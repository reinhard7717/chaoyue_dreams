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
        【V403.1 向量化性能重构版】
        - 核心优化 (本次修改):
          - [性能重构] 彻底移除了所有用于计算分数的 for 循环，改为调用新增的、完全向量化的 `_calculate_weighted_score` 辅助函数。
          - [效率提升] 将多个循环合并为少数几次高性能的NumPy矩阵运算，显著提升了计算效率，降低了内存分配开销。
        - 业务逻辑: 保持与V403.0版本完全一致，仅重构实现方式。
        """
        # print("        -> [进攻方案评估中心 V403.1 向量化性能重构版] 启动...") # 代码修改：更新版本号
        df = self.strategy.df_indicators
        atomic_states = self.strategy.atomic_states
        score_details_df = pd.DataFrame(index=df.index)
        scoring_params = get_params_block(self.strategy, 'four_layer_scoring_params')
        if not get_param_value(scoring_params.get('enabled'), True):
            return pd.Series(0.0, index=df.index), score_details_df
        
        all_scores_components = []
        default_series = pd.Series(0.0, index=df.index)
        
        # --- 步骤 1: 预计算特殊惩罚/奖励因子 ---
        df['LOW_21_D'] = df['low_D'].rolling(21).min()
        run_up_pct = (df['close_D'] - df['LOW_21_D']) / df['LOW_21_D']
        max_run_up_pct_for_bottom_reversal = 0.15
        bottom_reversal_penalty_multiplier = (1 - run_up_pct / max_run_up_pct_for_bottom_reversal).clip(lower=0, upper=1).fillna(1.0)
        special_multipliers = {"BOTTOM_REVERSAL": bottom_reversal_penalty_multiplier}

        # --- 步骤 2: 【代码修改】向量化计算第一层：环境与战备分 ---
        context_params = scoring_params.get('contextual_setup_scoring', {})
        context_score, score_details_df = self._calculate_weighted_score(
            context_params.get('positive_signals', {}),
            score_details_df,
            'SETUP_',
            special_multipliers
        )
        
        # --- NaN 探针与自我修复模块 (逻辑保持不变) ---
        if context_score.isnull().any():
            nan_mask = context_score.isnull()
            nan_dates = context_score[nan_mask].index
            print(f"  [严重警告-NaN探针] 在进攻层 'context_score' (SCORE_SETUP) 中检测到 {len(nan_dates)} 个 NaN 值！正在追溯源头...")
            for nan_date in nan_dates:
                culprit_signals = []
                for col in score_details_df.columns:
                    if col.startswith('SETUP_'):
                        if pd.isna(score_details_df.at[nan_date, col]):
                            culprit_signals.append(col.replace('SETUP_', ''))
                if culprit_signals:
                    print(f"    -> 日期: {nan_date.date()}, 罪魁祸首信号: {culprit_signals}。请检查这些信号的计算逻辑是否存在缺陷。")
            print("  [自我修复] 已将所有 NaN 值填充为 0，继续执行...")
            context_score.fillna(0, inplace=True)
            score_details_df.fillna(0, inplace=True)

        score_details_df['SCORE_SETUP'] = context_score
        all_scores_components.append(context_score)
        
        # --- 步骤 3: 【代码修改】向量化计算第二层：触发器事件分 ---
        trigger_params = scoring_params.get('trigger_event_scoring', {})
        trigger_score, score_details_df = self._calculate_weighted_score(
            trigger_params.get('positive_signals', {}),
            score_details_df,
            'TRIGGER_'
        )
        score_details_df['SCORE_TRIGGER'] = trigger_score
        all_scores_components.append(trigger_score)
        
        # --- 步骤 4: 【代码修改】向量化计算第三层：剧本协同分 ---
        playbook_params = scoring_params.get('playbook_synergy_scoring', {})
        playbook_score, score_details_df = self._calculate_weighted_score(
            playbook_params.get('positive_signals', {}),
            score_details_df,
            'PLAYBOOK_'
        )
        score_details_df['SCORE_PLAYBOOK_SYNERGY'] = playbook_score
        all_scores_components.append(playbook_score)
        
        # --- 步骤 5: 计算其他奖励分 (这些模块已部分优化，保持现状) ---
        strategic_bonus_score, score_details_df = self._apply_strategic_context_bonuses(score_details_df)
        all_scores_components.append(strategic_bonus_score)
        contextual_bonus_score, score_details_df = self._apply_contextual_bonus_score(score_details_df)
        all_scores_components.append(contextual_bonus_score)
        
        # 行业分计算已是向量化，保持不变
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
        
        # 【代码修改】向量化计算动态奖励分
        dynamic_score = pd.Series(0.0, index=df.index)
        dynamic_params = scoring_params.get('dynamic_scoring', {})
        if get_param_value(dynamic_params.get('enabled'), True):
            min_setup_score = get_param_value(dynamic_params.get('min_positional_score_for_dynamic'), 150)
            can_add_dynamic_score = (context_score >= min_setup_score).astype(float)
            
            dyn_signals_config = dynamic_params.get('positive_signals', {})
            dyn_score, score_details_df = self._calculate_weighted_score(
                dyn_signals_config,
                score_details_df,
                'DYN_'
            )
            # 将动态奖励分与条件相乘
            dynamic_score = dyn_score * can_add_dynamic_score
        all_scores_components.append(dynamic_score)
        
        # --- 步骤 6: 合成总进攻分 ---
        entry_score = sum(all_scores_components).fillna(0).astype(int)
        return entry_score, score_details_df.fillna(0)

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
        【V401.1 向量化性能重构版】进攻动能诊断大脑
        - 核心优化 (本次修改):
          - [性能重构] 彻底移除了原有的 for 循环，改为完全向量化的操作来生成诊断报告。
          - [效率提升] 通过布尔掩码和字符串拼接，一次性为所有日期生成报告，避免了逐行构建字典的巨大开销。
        - 业务逻辑: 保持与V401.0版本完全一致，仅重构实现方式。
        """
        # print("          -> [进攻动能诊断大脑 V401.1 向量化性能重构版] 启动，正在诊断分数动态...") # 代码修改：更新版本号
        # --- 步骤 1: 诊断总分(entry_score)的动态 (逻辑不变，已是向量化) ---
        score_change = entry_score.diff(1).fillna(0)
        score_accel = score_change.diff(1).fillna(0)
        scoring_params = get_params_block(self.strategy, 'four_layer_scoring_params')
        momentum_params = scoring_params.get('momentum_diagnostics_params', {})
        fading_score_threshold = get_param_value(momentum_params.get('fading_score_threshold'), 800)
        is_opportunity_fading = ((score_change > 0) & (score_accel < 0)) | (score_change <= 0)
        self.strategy.atomic_states['SCORE_DYN_OPPORTUNITY_FADING'] = is_opportunity_fading & (entry_score.shift(1) > fading_score_threshold)
        risk_score = self.strategy.df_indicators.get('risk_score', pd.Series(0.0, index=entry_score.index))
        risk_change = risk_score.diff(1).fillna(0)
        risk_accel = risk_change.diff(1).fillna(0)
        is_risk_escalating = (risk_change > 0) & (risk_accel > 0)
        self.strategy.atomic_states['SCORE_DYN_RISK_ESCALATING'] = is_risk_escalating
        
        # --- 步骤 2: 【代码修改】向量化生成用于调试的详细诊断报告 ---
        core_score = score_details_df.get('SCORE_SETUP', pd.Series(0.0, index=entry_score.index))
        core_score_change = core_score.diff(1).fillna(0)
        
        # 创建一个空的DataFrame来存储报告片段
        report_df = pd.DataFrame(index=entry_score.index)
        
        # 定义各种诊断条件
        stall_condition = (score_change <= 0) & (entry_score.shift(1) > 0)
        decel_condition = (score_change > 0) & (score_accel < 0)
        core_erosion_condition = (core_score_change < 0)
        divergence_condition = (core_score_change <= 0) & (score_change > 0) & (entry_score > 0)
        
        # 向量化地生成报告字符串
        report_df.loc[stall_condition, 'stall'] = "进攻停滞(总分变化: " + score_change[stall_condition].astype(int).astype(str) + ")"
        report_df.loc[decel_condition, 'deceleration'] = "进攻减速(加速度: " + score_accel[decel_condition].astype(int).astype(str) + ")"
        report_df.loc[core_erosion_condition, 'core_erosion'] = "核心侵蚀(战备分变化: " + core_score_change[core_erosion_condition].round(2).astype(str) + ")"
        report_df.loc[divergence_condition, 'divergence'] = "结构性背离(总分虚高)"
        
        # 将报告片段DataFrame转换为字典的Series
        # .apply() 在这里是最高效的方式，因为它操作的是已经计算好的、稀疏的字符串数据
        diagnostics = report_df.apply(lambda row: {k: v for k, v in row.dropna().items()}, axis=1)
        
        return diagnostics

    def _calculate_weighted_score(self, signals_config: Dict, score_details_df: pd.DataFrame, prefix: str, special_multipliers: Dict = None) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【V403.1 新增】向量化加权分数计算辅助函数
        - 核心职责: 替代原有的for循环，通过向量化操作一次性计算一个层级的总分。
        - 性能优势: 避免了在循环中反复创建和累加Pandas Series的巨大开销，将计算复杂度从 O(N*M) 降低到 O(N)，其中N是信号数量，M是数据长度。
        - 实现方式:
          1. 将所有信号的Series和权重值分别提取出来。
          2. 将信号Series列表转换为一个2D NumPy数组。
          3. 使用NumPy广播机制，将信号数组与权重数组高效相乘。
          4. 对结果沿信号维度求和，得到最终的总分Series。
        """
        atomic_states = self.strategy.atomic_states
        df_index = self.strategy.df_indicators.index
        default_series = pd.Series(0.0, index=df_index)
        
        signal_names = []
        score_weights = []
        signal_series_list = []

        # 步骤1: 收集所有有效的信号和权重
        for signal_name, score in signals_config.items():
            if signal_name.startswith("说明"):
                continue
            
            # 根据信号来源获取Series
            if prefix == 'TRIGGER_':
                signal_series = self.strategy.trigger_events.get(signal_name, pd.Series(False, index=df_index))
            elif prefix == 'PLAYBOOK_':
                signal_series = self.strategy.playbook_states.get(signal_name, pd.Series(False, index=df_index))
            else: # 默认为SETUP
                signal_series = atomic_states.get(signal_name, pd.Series(False, index=df_index))

            signal_names.append(signal_name)
            score_weights.append(score)
            signal_series_list.append(signal_series.astype(float))

        if not signal_series_list:
            return default_series, score_details_df

        # 步骤2: 将Series列表转换为2D NumPy数组
        signals_array = np.stack([s.values for s in signal_series_list], axis=0)
        
        # 步骤3: 应用特殊乘数（如底部反转惩罚）
        weights_array = np.array(score_weights, dtype=np.float32).reshape(-1, 1)
        if special_multipliers:
            for multiplier_key, multiplier_series in special_multipliers.items():
                for i, name in enumerate(signal_names):
                    if multiplier_key in name:
                        # 直接在NumPy层面应用乘数
                        signals_array[i, :] *= multiplier_series.values

        # 步骤4: 向量化计算每个信号的贡献分数
        bonus_amounts_array = signals_array * weights_array
        
        # 步骤5: 计算总分
        total_score_array = np.sum(bonus_amounts_array, axis=0)
        total_score_series = pd.Series(total_score_array, index=df_index)

        # 步骤6: 更新score_details_df (用于调试)
        for i, signal_name in enumerate(signal_names):
            # 只记录有贡献的信号，避免DataFrame过于稀疏
            if np.any(bonus_amounts_array[i] > 0):
                score_details_df[f"{prefix}{signal_name}"] = bonus_amounts_array[i]
        
        return total_score_series, score_details_df













