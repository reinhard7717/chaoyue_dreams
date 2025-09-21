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
        【V403.0 NaN探针与自修复版】
        - 核心升级 (本次修改):
          - [新增探针] 在计算完 `context_score` (即 SCORE_SETUP) 后，立即植入一个“NaN探针”。如果检测到NaN，它会打印出详细的错误报告，精确指出是哪个上游信号在哪一天产生了NaN。
          - [自我修复] 在探针诊断后，立即使用 `.fillna(0)` 对所有分数组件进行“自我修复”，从根本上阻止NaN污染下游模块。
        - 收益: 不仅修复了崩溃，更重要的是建立了强大的自我诊断能力，极大地提升了系统的可维护性。
        """
        # print("        -> [进攻方案评估中心 V403.0 NaN探针与自修复版] 启动...") # 修改: 更新版本号
        df = self.strategy.df_indicators
        atomic_states = self.strategy.atomic_states
        score_details_df = pd.DataFrame(index=df.index)
        scoring_params = get_params_block(self.strategy, 'four_layer_scoring_params')
        if not get_param_value(scoring_params.get('enabled'), True):
            return pd.Series(0.0, index=df.index), score_details_df
        all_scores_components = []
        default_series = pd.Series(0.0, index=df.index)
        
        # ... (反过热机制代码保持不变) ...
        df['LOW_21_D'] = df['low_D'].rolling(21).min()
        run_up_pct = (df['close_D'] - df['LOW_21_D']) / df['LOW_21_D']
        max_run_up_pct_for_bottom_reversal = 0.15
        bottom_reversal_penalty_multiplier = (1 - run_up_pct / max_run_up_pct_for_bottom_reversal).clip(lower=0, upper=1).fillna(1.0)

        # --- 步骤 2: 计算【第一层：环境与战备分】 ---
        context_params = scoring_params.get('contextual_setup_scoring', {})
        context_score = pd.Series(0.0, index=df.index)
        if get_param_value(context_params.get('enabled'), True):
            for signal_name, score in context_params.get('positive_signals', {}).items():
                if signal_name.startswith("说明"):
                    continue
                signal_series = atomic_states.get(signal_name, pd.Series(False, index=df.index))
                if signal_series.any():
                    bonus_amount = signal_series.astype(float) * score # 注意：此处先不 fillna，以暴露问题
                    if "BOTTOM_REVERSAL" in signal_name:
                        bonus_amount *= bottom_reversal_penalty_multiplier
                    context_score += bonus_amount
                    score_details_df[f"SETUP_{signal_name}"] = bonus_amount
        
        # --- 新增开始：NaN 探针与自我修复模块 ---
        if context_score.isnull().any():
            nan_mask = context_score.isnull()
            nan_dates = context_score[nan_mask].index
            print(f"  [严重警告-NaN探针] 在进攻层 'context_score' (SCORE_SETUP) 中检测到 {len(nan_dates)} 个 NaN 值！正在追溯源头...")
            for nan_date in nan_dates:
                culprit_signals = []
                # 找出所有在这一天贡献了 NaN 的上游信号
                for col in score_details_df.columns:
                    if col.startswith('SETUP_'):
                        if pd.isna(score_details_df.at[nan_date, col]):
                            culprit_signals.append(col.replace('SETUP_', ''))
                if culprit_signals:
                    print(f"    -> 日期: {nan_date.date()}, 罪魁祸首信号: {culprit_signals}。请检查这些信号的计算逻辑是否存在缺陷。")
            
            # 自我修复：用0填充所有 NaN，防止下游崩溃
            print("  [自我修复] 已将所有 NaN 值填充为 0，继续执行...")
            context_score.fillna(0, inplace=True)
            score_details_df.fillna(0, inplace=True)
        # --- 新增结束 ---

        score_details_df['SCORE_SETUP'] = context_score
        all_scores_components.append(context_score)
        
        # ... (后续代码保持不变，但现在它们接收的是无NaN的数据) ...
        trigger_params = scoring_params.get('trigger_event_scoring', {})
        trigger_score = pd.Series(0.0, index=df.index)
        if get_param_value(trigger_params.get('enabled'), True):
            for trigger_name, score in trigger_params.get('positive_signals', {}).items():
                if trigger_name.startswith("说明"): continue
                trigger_series = trigger_events.get(trigger_name, pd.Series(False, index=df.index))
                if trigger_series.any():
                    bonus_amount = trigger_series.fillna(0).astype(float) * score
                    trigger_score += bonus_amount
                    score_details_df[f"TRIGGER_{trigger_name}"] = bonus_amount
        score_details_df['SCORE_TRIGGER'] = trigger_score
        all_scores_components.append(trigger_score)
        
        playbook_params = scoring_params.get('playbook_synergy_scoring', {})
        playbook_score = pd.Series(0.0, index=df.index)
        if get_param_value(playbook_params.get('enabled'), True):
            for playbook_name, score in playbook_params.get('positive_signals', {}).items():
                if playbook_name.startswith("说明"): continue
                playbook_series = self.strategy.playbook_states.get(playbook_name, pd.Series(False, index=df.index))
                if playbook_series.any():
                    bonus_amount = playbook_series.fillna(0).astype(float) * score
                    playbook_score += bonus_amount
                    score_details_df[f"PLAYBOOK_{playbook_name}"] = bonus_amount
        score_details_df['SCORE_PLAYBOOK_SYNERGY'] = playbook_score
        all_scores_components.append(playbook_score)
        
        strategic_bonus_score, score_details_df = self._apply_strategic_context_bonuses(score_details_df)
        all_scores_components.append(strategic_bonus_score)
        contextual_bonus_score, score_details_df = self._apply_contextual_bonus_score(score_details_df)
        all_scores_components.append(contextual_bonus_score)
        
        industry_score = pd.Series(0.0, index=df.index)
        industry_params = scoring_params.get('industry_lifecycle_scoring_params', {})
        if get_param_value(industry_params.get('enabled'), True):
            score_markup = atomic_states.get('SCORE_INDUSTRY_MARKUP', default_series)
            score_preheat = atomic_states.get('SCORE_INDUSTRY_PREHEAT', default_series)
            markup_weight = industry_params.get('markup_weight', 1.0)
            preheat_weight = industry_params.get('preheat_weight', 0.8)
            bonus_multiplier = industry_params.get('bonus_multiplier', 400)
            positive_industry_factor = (score_markup * markup_weight + score_preheat * preheat_weight)
            industry_bonus = positive_industry_factor * bonus_multiplier
            if (industry_bonus > 1).any():
                industry_score += industry_bonus
                score_details_df["BONUS_INDUSTRY_LIFECYCLE"] = industry_bonus
        all_scores_components.append(industry_score)
        
        dynamic_score = pd.Series(0.0, index=df.index)
        dynamic_params = scoring_params.get('dynamic_scoring', {})
        if get_param_value(dynamic_params.get('enabled'), True):
            min_setup_score = get_param_value(dynamic_params.get('min_positional_score_for_dynamic'), 150)
            can_add_dynamic_score = context_score >= min_setup_score
            for signal_name, score in dynamic_params.get('positive_signals', {}).items():
                if signal_name.startswith("说明"): continue
                signal_series = atomic_states.get(signal_name, pd.Series(0.0, index=df.index))
                if (signal_series > 0).any():
                    bonus_amount = signal_series.fillna(0.0) * score * can_add_dynamic_score.astype(float)
                    dynamic_score += bonus_amount
                    score_details_df[f"DYN_{signal_name}"] = bonus_amount
        all_scores_components.append(dynamic_score)
        
        # --- 步骤 6: 合成总进攻分 ---
        entry_score = sum(all_scores_components).fillna(0).astype(int)
        return entry_score, score_details_df.fillna(0) # 返回前再次确认填充，确保万无一失

    def _apply_strategic_context_bonuses(self, score_details_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【V401.2 逻辑修复版】战略背景奖励模块
        - 核心修复: 修改函数签名，不再接收并修改 entry_score，而是返回一个独立的 bonus_score Series。
        """
        bonus_score = pd.Series(0.0, index=score_details_df.index) # 代码初始化独立的奖励分Series
        scoring_params = get_params_block(self.strategy, 'four_layer_scoring_params')
        atomic_states = self.strategy.atomic_states
        default_series = pd.Series(False, index=bonus_score.index)
        # 1. 处理周线战略背景加分 (strategic_context_scoring)
        strategic_params = scoring_params.get('strategic_context_scoring', {})
        if get_param_value(strategic_params.get('enabled'), False):
            strategic_map = {
                'CONTEXT_STRATEGIC_BULLISH_W': 'bullish_bonus',
                'CONTEXT_STRATEGIC_IGNITION_W': 'ignition_bonus',
                'CONTEXT_TREND_HEALTH_STRONG_W': 'trend_health_bonus',
                'CONTEXT_NEAR_52W_HIGH_W': 'breakout_eve_bonus',
                'CONTEXT_PSYCH_REVERSAL_BULLISH_W': 'reversal_confirm_bonus',
                'CONTEXT_CHIP_LONG_TERM_ACCUMULATION_D': 'long_term_chip_accumulation_bonus',
                'CONTEXT_CHIP_LONG_TERM_ACCEL_ACCUMULATION_D': 'long_term_chip_accel_accumulation_bonus',
                'CONTEXT_CHIP_LONG_TERM_HEALTH_IMPROVING_D': 'long_term_chip_health_improving_bonus',
                'CONTEXT_CHIP_LONG_TERM_DIVERGENCE_D': 'long_term_chip_divergence_penalty',
                'CONTEXT_CHIP_LONG_TERM_ACCEL_DIVERGENCE_D': 'long_term_chip_accel_divergence_penalty',
            }
            for signal_name, config_key in strategic_map.items():
                signal_series = atomic_states.get(signal_name, default_series)
                if signal_series.any():
                    score_value = get_param_value(strategic_params.get(config_key), 0)
                    if score_value != 0:
                        bonus_amount = signal_series.fillna(0).astype(float) * score_value
                        bonus_score += bonus_amount # 代码累加到独立的 bonus_score
                        score_details_df[f"STRATEGIC_{signal_name}"] = bonus_amount
        # 2. 处理日线长周期筹码战略背景加分 (chip_context_scoring)
        chip_context_params = scoring_params.get('chip_context_scoring', {})
        if get_param_value(chip_context_params.get('enabled'), False):
            signal_name = 'CONTEXT_CHIP_STRATEGIC_GATHERING'
            signal_series = atomic_states.get(signal_name, default_series)
            if signal_series.any():
                score_value = get_param_value(chip_context_params.get('strategic_gathering_bonus'), 0)
                if score_value != 0:
                    bonus_amount = signal_series.fillna(0).astype(float) * score_value
                    bonus_score += bonus_amount # 代码累加到独立的 bonus_score
                    score_details_df[f"STRATEGIC_{signal_name}"] = bonus_amount
        return bonus_score, score_details_df # 代码返回独立的 bonus_score

    def _apply_contextual_bonus_score(self, score_details_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【V4.2 逻辑修复版】战术环境奖励模块
        - 核心修复: 修改函数签名，不再接收并修改 entry_score，而是返回一个独立的 bonus_score Series。
        """
        bonus_score = pd.Series(0.0, index=score_details_df.index) # 代码初始化独立的奖励分Series
        bonus_params = get_params_block(self.strategy, 'contextual_bonus_params')
        if not get_param_value(bonus_params.get('enabled'), False):
            return bonus_score, score_details_df
        bonus_rules = bonus_params.get('bonuses', [])
        for rule in bonus_rules:
            state_name = rule.get('if_state')
            bonus_signal_name = rule.get('signal_name')
            if not (state_name and bonus_signal_name) or rule.get('deprecated', False):
                continue
            condition = self.strategy.atomic_states.get(state_name, pd.Series(False, index=bonus_score.index)).shift(1).fillna(False)
            if not condition.any():
                continue
            if rule.get('decay_model', False):
                max_bonus = rule.get('max_bonus_score', 0)
                decay_days = rule.get('decay_days', 1)
                if max_bonus <= 0 or decay_days <= 0 or not condition.any():
                    continue
                influence_series = self.strategy.cognitive_intel._create_decaying_influence_series(condition, decay_days)
                bonus_amount = influence_series * max_bonus
                bonus_score += bonus_amount # 代码累加到独立的 bonus_score
                if bonus_signal_name not in score_details_df.columns:
                    score_details_df[bonus_signal_name] = 0.0
                score_details_df[bonus_signal_name] += bonus_amount
            else:
                bonus_value = rule.get('add_score', 0)
                if condition.any() and bonus_value != 0:
                    bonus_amount = condition.astype(float) * bonus_value
                    bonus_score += bonus_amount # 代码累加到独立的 bonus_score
                    score_details_df[bonus_signal_name] = bonus_amount
        return bonus_score, score_details_df # 代码返回独立的 bonus_score

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















