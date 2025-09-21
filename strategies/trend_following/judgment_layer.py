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
        【V2.3 · 健壮性修复版】生成人类可读的信号摘要。
        - 核心修复 (本次修改):
          - [健壮性] 修复了因 `reset_index()` 行为不确定而导致的 `KeyError: 'index'` 崩溃问题。
          - [解决方案] 不再硬编码 `groupby('index')`，而是通过 `long_df.columns[0]` 动态获取由 `reset_index()` 生成的日期列的实际名称，确保无论原始索引是否有名称，代码都能正确分组。
        - 业务逻辑: 保持与V2.2版本完全一致，仅修复实现上的bug。
        """
        # 加载信号与中文名的映射字典
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        # 定义已知的前缀列表
        prefixes_to_strip = ['SETUP_', 'TRIGGER_', 'PLAYBOOK_', 'DYN_', 'STRATEGIC_', 'BONUS_']

        def process_details_df(details_df, prefix_list):
            """辅助函数，用于向量化处理一个分数详情DataFrame"""
            if details_df.empty:
                return pd.Series(dtype=object)
            
            # 1. 宽表转长表，并过滤无效分数
            # ignore_index=False 保留原始索引，reset_index() 将其转换为列
            long_df = details_df.melt(ignore_index=False, var_name='signal', value_name='score').reset_index()
            long_df = long_df[long_df['score'] > 0].copy()
            if long_df.empty:
                return pd.Series(dtype=object)

            # 动态获取由 reset_index() 生成的日期列的名称。
            # 这使得代码不再依赖于 'index' 这个不确定的默认名称，从而修复了KeyError。
            date_col_name = long_df.columns[0]
            # print(f"调试信息: process_details_df 中动态获取的日期列名为: '{date_col_name}'")

            # 2. 向量化剥离前缀
            long_df['base_signal'] = long_df['signal']
            for prefix in prefix_list:
                long_df['base_signal'] = long_df['base_signal'].str.removeprefix(prefix)

            # 3. 向量化映射中文名
            cn_name_map = {k: v.get('cn_name', k) for k, v in score_map.items()}
            long_df['cn_name'] = long_df['base_signal'].map(cn_name_map).fillna(long_df['base_signal'])
            
            # 4. 向量化生成摘要字符串
            long_df['summary_str'] = long_df['cn_name'] + " (" + long_df['score'].astype(int).astype(str) + ")"
            
            # 5. 按日期分组并聚合为列表
            # 使用动态获取的日期列名进行分组。
            return long_df.groupby(date_col_name)['summary_str'].apply(list)

        # --- 调用向量化辅助函数处理进攻和风险信号 ---
        offense_summaries = process_details_df(score_details_df, prefixes_to_strip)
        risk_summaries = process_details_df(risk_details_df, []) # 风险信号通常没有前缀

        # 合并进攻和风险摘要
        summary_df = pd.DataFrame({
            'offense': offense_summaries,
            'risk': risk_summaries
        }).reindex(self.strategy.df_indicators.index) # 确保索引与主DataFrame对齐

        # 将两列转换为最终的字典格式
        # .apply 在这里是最高效的方式，因为它操作的是已经聚合好的小数据
        final_summaries = summary_df.apply(
            lambda row: {
                'offense': row['offense'] if isinstance(row['offense'], list) else [],
                'risk': row['risk'] if isinstance(row['risk'], list) else []
            },
            axis=1
        )
        
        return final_summaries

    def make_final_decisions(self, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame):
        """
        【V502.1 职责净化版】
        - 核心架构升级: 废除“否决票”机制，采用“风险惩罚分”模型。
        - 决策逻辑简化: 最终决策逻辑简化为 `最终得分 = 进攻分 - 风险惩罚分`。
        - 职责净化: 移除了对 `_generate_exit_triggers` 的调用，硬性离场决策完全交由上层模块处理。
        """
        # print("    --- [最高作战指挥部 V502.1 职责净化版] 启动... ---")
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
        【V318.1 信号源修复版】动态力学战术矩阵
        - 核心重构 (本次修改):
          - [逻辑修正] 修复了对动态力学信号的调用错误。原代码消费已废弃的 `SCORE_FV_*` 信号，
                        导致该战术矩阵失效。新代码已修正为消费由 `DynamicMechanicsEngine` V3.0+ 
                        生成的、正确的 `SCORE_DYN_*` 终极信号。
          - `FORCE_ATTACK` (强攻): 由 `SCORE_DYN_BULLISH_RESONANCE_S` (动态看涨共振) 驱动。
          - `AVOID` (规避): 由 `SCORE_DYN_BEARISH_RESONANCE_S` (动态看跌共振) 驱动。
        - 收益: 恢复了动态战术矩阵的核心功能，使其能根据最新的、高质量的力学情报做出正确的战术响应。
        """
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 全面使用新版S级数值化信号 ---
        # 获取动态看涨共振S级分数，代表“纯粹的进攻”
        # 修正了信号名称，从 SCORE_FV_OFFENSIVE_RESONANCE_S 改为 SCORE_DYN_BULLISH_RESONANCE_S
        offensive_resonance_score = atomic.get('SCORE_DYN_BULLISH_RESONANCE_S', default_score)
        # 获取动态看跌共振S级分数，代表“高位滞涨/出货”风险
        # 修正了信号名称，从 SCORE_FV_RISK_EXPANSION_S 改为 SCORE_DYN_BEARISH_RESONANCE_S
        risk_expansion_score = atomic.get('SCORE_DYN_BEARISH_RESONANCE_S', default_score)
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
        【V400.3 行业协同版】计算风险惩罚分
        - 核心升级 (本次修改):
          - [新增] 引入了对行业生命周期风险的量化惩罚。
          - 当行业处于“高位滞涨”或“下跌通道”时，会根据其风险置信度分数，按比例增加风险惩罚分。
        - 收益: 实现了对宏观行业风险的精准量化和主动规避。
        """
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        default_series = pd.Series(0.0, index=df.index)
        def get_clipped_score(signal_name):
            """辅助函数：获取信号分并确保其非负"""
            return atomic.get(signal_name, default_series).clip(lower=0)
        # --- 风险1: 绝对否决信号 (权重: 300) ---
        veto_params = get_params_block(self.strategy, 'absolute_veto_params')
        if get_param_value(veto_params.get('enabled'), True):
            mitigation_rules = get_param_value(veto_params.get('mitigation_rules'), {})
            veto_signals = get_param_value(veto_params.get('veto_signals'), [])
            for signal_name in veto_signals:
                risk_score = get_clipped_score(signal_name)
                if risk_score.sum() == 0: continue # 优化：如果没有信号，直接跳过
                # 处理风险缓解
                if signal_name in mitigation_rules:
                    mitigators = mitigation_rules[signal_name].get('mitigated_by', [])
                    mitigator_score = pd.Series(0.0, index=df.index)
                    if mitigators:
                        mitigator_scores = [get_clipped_score(m_signal) for m_signal in mitigators]
                        if mitigator_scores:
                            mitigator_score = pd.Series(np.maximum.reduce([s.values for s in mitigator_scores if not s.empty]), index=df.index)
                    
                    net_risk_score = risk_score * (1 - mitigator_score)
                    df['risk_penalty_score'] += net_risk_score * 300
                else:
                    df['risk_penalty_score'] += risk_score * 300
        # --- 风险2: 风险分高于进攻分 (权重: 100) ---
        risk_overrides_score = (df['risk_score'] > df['entry_score']).astype(float)
        is_in_ascent_phase = get_clipped_score('SCORE_STRUCTURE_MAIN_UPTREND_WAVE_S')
        net_risk_score = risk_overrides_score * (1 - is_in_ascent_phase)
        df['risk_penalty_score'] += net_risk_score * 100
        # --- 风险3: 周线战略顶层风险 (权重: 300 for topping, 100 for bearish) ---
        df['risk_penalty_score'] += get_clipped_score('CONTEXT_STRATEGIC_TOPPING_RISK_W') * 300
        df['risk_penalty_score'] += get_clipped_score('CONTEXT_STRATEGIC_BEARISH_W') * 100
        # --- 风险4: 顶层认知风险 (权重: 200-300) ---
        df['risk_penalty_score'] += get_clipped_score('COGNITIVE_ULTIMATE_BEARISH_CONFIRMATION_S') * 300
        df['risk_penalty_score'] += get_clipped_score('COGNITIVE_SCORE_BREAKDOWN_RESONANCE_S') * 300
        df['risk_penalty_score'] += get_clipped_score('COGNITIVE_SCORE_TOP_REVERSAL_RESONANCE_S') * 250
        df['risk_penalty_score'] += get_clipped_score('COGNITIVE_SCORE_MULTI_DIMENSIONAL_DIVERGENCE_S') * 200
        df['risk_penalty_score'] += get_clipped_score('COGNITIVE_SCORE_TREND_FATIGUE_RISK') * 150
        # --- 风险5: 各情报层S级看跌共振风险 (权重: 150) ---
        df['risk_penalty_score'] += get_clipped_score('SCORE_BEHAVIOR_BEARISH_RESONANCE_S') * 150
        df['risk_penalty_score'] += get_clipped_score('SCORE_CHIP_BEARISH_RESONANCE_S') * 150
        df['risk_penalty_score'] += get_clipped_score('SCORE_DYN_BEARISH_RESONANCE_S') * 150
        df['risk_penalty_score'] += get_clipped_score('SCORE_FF_BEARISH_RESONANCE_S') * 150
        df['risk_penalty_score'] += get_clipped_score('SCORE_STRUCTURE_BEARISH_RESONANCE_S') * 150
        df['risk_penalty_score'] += get_clipped_score('SCORE_FOUNDATION_BEARISH_RESONANCE_S') * 150
        # --- 风险6: 微观结构风险 (权重: 250-280) ---
        df['risk_penalty_score'] += get_clipped_score('COGNITIVE_SCORE_RISK_POWER_SHIFT_TO_RETAIL') * 280
        df['risk_penalty_score'] += get_clipped_score('COGNITIVE_SCORE_RISK_MAIN_FORCE_CONVICTION_WEAKENING') * 250
        # --- 风险18: 行业生命周期风险 (权重由配置决定) ---
        industry_params = get_params_block(self.strategy, 'four_layer_scoring_params', {}).get('industry_lifecycle_scoring_params', {})
        if industry_params.get('enabled', False):
            penalty_multiplier = industry_params.get('penalty_multiplier', 600)
            # 获取滞涨和下跌阶段的数值化置信度分数
            score_stagnation = get_clipped_score('SCORE_INDUSTRY_STAGNATION')
            score_downtrend = get_clipped_score('SCORE_INDUSTRY_DOWNTREND')
            # 获取惩罚权重
            stagnation_weight = industry_params.get('stagnation_penalty_weight', 1.2)
            downtrend_weight = industry_params.get('downtrend_penalty_weight', 1.5)
            # 计算惩罚分，与置信度成正比
            stagnation_penalty = score_stagnation * stagnation_weight * penalty_multiplier
            downtrend_penalty = score_downtrend * downtrend_weight * penalty_multiplier
            df['risk_penalty_score'] += stagnation_penalty
            df['risk_penalty_score'] += downtrend_penalty
            # 记录到 risk_details_df 以便调试
            if hasattr(self.strategy, 'risk_details_df'):
                 if (stagnation_penalty > 0).any():
                     self.strategy.risk_details_df['RISK_INDUSTRY_STAGNATION'] = stagnation_penalty
                 if (downtrend_penalty > 0).any():
                     self.strategy.risk_details_df['RISK_INDUSTRY_DOWNTREND'] = downtrend_penalty
    
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














