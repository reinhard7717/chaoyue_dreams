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
        【V500.0 哲学升维版 - 反转优先】
        - 核心重构 (本次修改):
          - [战略转移] 彻底重构计分体系，从“重视共振”转向“重视反转”。
          - [双核驱动] 将进攻分拆分为“反转进攻分”和“共振进攻分”两个独立的引擎。
          - [反转优先] “反转进攻分”由全新的“反转可靠性”超级信号驱动，并赋予极高权重。
          - [共振降权] “共振进攻分”的分数权重被全面下调，作为趋势中段的辅助加分项，而非决策主力。
        - 收益: 策略的核心逻辑与“买在分歧，卖在一致”的A股实战哲学深度对齐，旨在捕捉更安全、赔率更高的反转初期机会。
        """
        print("        -> [进攻方案评估中心 V500.0 反转优先版] 启动...") # 修改: 更新版本号
        df = self.strategy.df_indicators
        atomic_states = self.strategy.atomic_states
        score_details_df = pd.DataFrame(index=df.index)
        scoring_params = get_params_block(self.strategy, 'four_layer_scoring_params')
        if not get_param_value(scoring_params.get('enabled'), True):
            return pd.Series(0.0, index=df.index), score_details_df
        
        default_series = pd.Series(0.0, index=df.index)
        
        # --- 步骤 1: 【新增】计算“反转进攻分” (Reversal Offense Score) ---
        reversal_params = scoring_params.get('reversal_offense_scoring', {})
        reversal_score, score_details_df = self._calculate_weighted_score(
            reversal_params.get('positive_signals', {}),
            score_details_df,
            'REVERSAL_'
        )
        
        # --- 步骤 2: 计算“共振进攻分” (Resonance Offense Score) ---
        # 注意：这里的权重已在配置文件中全面下调
        resonance_params = scoring_params.get('resonance_offense_scoring', {})
        resonance_score, score_details_df = self._calculate_weighted_score(
            resonance_params.get('positive_signals', {}),
            score_details_df,
            'RESONANCE_'
        )

        # --- 步骤 3: 计算剧本、触发器等其他分数 (逻辑简化) ---
        playbook_params = scoring_params.get('playbook_synergy_scoring', {})
        playbook_score, score_details_df = self._calculate_weighted_score(
            playbook_params.get('positive_signals', {}),
            score_details_df,
            'PLAYBOOK_'
        )
        
        trigger_params = scoring_params.get('trigger_event_scoring', {})
        trigger_score, score_details_df = self._calculate_weighted_score(
            trigger_params.get('positive_signals', {}),
            score_details_df,
            'TRIGGER_'
        )

        # --- 步骤 4: 合成总进攻分 ---
        # 总分 = 反转分 + 共振分 + 剧本分 + 触发器分
        entry_score = (reversal_score + resonance_score + playbook_score + trigger_score).fillna(0).astype(int)
        
        # --- 记录各部分汇总分，便于调试 ---
        score_details_df['SCORE_REVERSAL_OFFENSE'] = reversal_score
        score_details_df['SCORE_RESONANCE_OFFENSE'] = resonance_score
        score_details_df['SCORE_PLAYBOOK_SYNERGY'] = playbook_score
        score_details_df['SCORE_TRIGGER'] = trigger_score

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

    def _get_human_readable_summary(self, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame) -> pd.Series:
        """
        【V2.6 · 翻译官逻辑修复版】生成人类可读的信号摘要。
        - 核心修复 (本次修改):
          - [智能识别] 进一步增强了 get_base_signal 辅助函数，增加了对 'TRIGGER_' 和 'PLAYBOOK_' 前缀的特殊处理。
        - 收益: 彻底解决了因触发器和剧本信号名被错误剥离，导致报告层找不到其定义的根本问题。
        """
        # 加载信号与中文名的映射字典
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        
        # 定义所有汇总分数的名称，这些名称不应被剥离前缀
        summary_score_cols = [
            'SCORE_SETUP', 'SCORE_TRIGGER', 'SCORE_PLAYBOOK_SYNERGY',
            'SCORE_REVERSAL_OFFENSE', 'SCORE_RESONANCE_OFFENSE'
        ]
        # 定义需要剥离的前缀列表
        prefixes_to_strip = ['SETUP_', 'TRIGGER_', 'PLAYBOOK_', 'DYN_', 'STRATEGIC_', 'BONUS_', 'REVERSAL_', 'RESONANCE_']

        def process_details_df(details_df, prefix_list):
            """辅助函数，用于向量化处理一个分数详情DataFrame"""
            if details_df.empty:
                return pd.Series(dtype=object)
            
            long_df = details_df.melt(ignore_index=False, var_name='signal', value_name='score').reset_index()
            long_df = long_df[long_df['score'] > 0].copy()
            if long_df.empty:
                return pd.Series(dtype=object)

            date_col_name = long_df.columns[0]
            
            def get_base_signal(signal_name):
                # --- 修改开始：增加对TRIGGER和PLAYBOOK的特殊处理 ---
                # 如果是汇总分、触发器或剧本，直接返回原名，因为它们在字典中的键就是全名
                if (signal_name in summary_score_cols or 
                    signal_name.startswith('TRIGGER_') or 
                    signal_name.startswith('PLAYBOOK_')):
                    return signal_name
                # --- 修改结束 ---
                
                # 否则，正常进行前缀剥离
                base_name = signal_name
                for prefix in prefix_list:
                    base_name = base_name.removeprefix(prefix)
                return base_name

            # 应用这个智能剥离函数
            long_df['base_signal'] = long_df['signal'].apply(get_base_signal)

            # 向量化映射中文名
            cn_name_map = {k: v.get('cn_name', k) for k, v in score_map.items()}
            long_df['cn_name'] = long_df['base_signal'].map(cn_name_map).fillna(long_df['base_signal'])
            
            # 向量化生成摘要字符串
            long_df['summary_str'] = long_df['cn_name'] + " (" + long_df['score'].astype(int).astype(str) + ")"
            
            # 按日期分组并聚合为列表
            return long_df.groupby(date_col_name)['summary_str'].apply(list)

        # --- 调用向量化辅助函数处理进攻和风险信号 ---
        offense_summaries = process_details_df(score_details_df, prefixes_to_strip)
        risk_summaries = process_details_df(risk_details_df, []) # 风险信号通常没有前缀

        # 合并进攻和风险摘要
        summary_df = pd.DataFrame({
            'offense': offense_summaries,
            'risk': risk_summaries
        }).reindex(self.strategy.df_indicators.index)

        # 将两列转换为最终的字典格式
        final_summaries = summary_df.apply(
            lambda row: {
                'offense': row['offense'] if isinstance(row['offense'], list) else [],
                'risk': row['risk'] if isinstance(row['risk'], list) else []
            },
            axis=1
        )
        
        return final_summaries

    def _calculate_weighted_score(self, signals_config: Dict, score_details_df: pd.DataFrame, prefix: str, special_multipliers: Dict = None) -> Tuple[pd.Series, pd.DataFrame]:
            """
            【V403.3 数据净化版】向量化加权分数计算辅助函数
            - 核心修复 (本次修改):
            - [数据净化] 修正了向 score_details_df 中写入列名的逻辑。现在写入的是原始信号名 (signal_name)，而不是带前缀的内部名称。
            - 收益: 从根源上解决了下游模块无法识别带前缀信号名的问题，确保了 score_details_df 在整个系统中的格式一致性和可用性。
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
                
                score_value = 0
                if isinstance(score, dict):
                    # 如果值是一个字典，从中提取 'score' 键的值
                    score_value = score.get('score', 0)
                elif isinstance(score, (int, float)):
                    # 如果值直接是数字，直接使用（兼容旧配置）
                    score_value = score

                # 根据信号来源获取Series
                if prefix == 'TRIGGER_':
                    signal_series = self.strategy.trigger_events.get(signal_name, pd.Series(False, index=df_index))
                elif prefix == 'PLAYBOOK_':
                    signal_series = self.strategy.playbook_states.get(signal_name, pd.Series(False, index=df_index))
                else: # 默认为反转或共振信号
                    signal_series = atomic_states.get(signal_name, pd.Series(False, index=df_index))

                signal_names.append(signal_name)
                score_weights.append(score_value)
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
                    # 确保 score_details_df 中的列名是全系统可识别的标准名称
                    # 如果一个信号在多个计分引擎中都贡献了分数，这里会累加
                    if signal_name not in score_details_df.columns:
                        score_details_df[signal_name] = 0.0
                    score_details_df[signal_name] += bonus_amounts_array[i]
            
            return total_score_series, score_details_df









