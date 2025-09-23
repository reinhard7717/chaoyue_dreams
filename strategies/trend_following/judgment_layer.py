# 文件: strategies/trend_following/judgment_layer.py
# 统合判断层 (V404.2 - 逻辑净化版)
import pandas as pd
import numpy as np
from .utils import get_params_block, get_param_value

class JudgmentLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def make_final_decisions(self, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame):
        """
        【V504.0 风险会计师版】
        - 核心升级 (本次修改):
          - [新增探针] 新增了 `_deploy_risk_accountant_probe` 探针，用于在调试模式下，精确解剖“风险惩罚分”的构成。
          - [重构数据流] 修改了 `_calculate_risk_penalty_score` 的逻辑，使其返回一个包含所有风险项贡献分的DataFrame，供新探针消费。
        - 收益: 彻底解决了风险惩罚分计算过程“黑箱化”的问题，为下一步精确优化风险模型提供了必需的可观测性。
        """
        
        print("    --- [最高作战指挥部 V504.0 风险会计师版] 启动... ---")
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        
        # --- 步骤 1: 计算风险惩罚分，并获取其详细构成 ---
        # 接收返回的风险构成DataFrame
        risk_components_df = self._calculate_risk_penalty_score()
        df['risk_penalty_score'] = risk_components_df.sum(axis=1)

        # --- 步骤 2: 获取亢奋风险并计算衰减因子 ---
        p_judge = get_params_block(self.strategy, 'four_layer_scoring_params').get('judgment_params', {})
        euphoria_risk_score = atomic.get('COGNITIVE_SCORE_RISK_EUPHORIC_ACCELERATION', pd.Series(0.0, index=df.index))
        attenuation_factor = (euphoria_risk_score * get_param_value(p_judge.get('euphoria_attenuation_multiplier'), 2.0)).clip(0, 1)

        # --- 步骤 3: 获取动态战术动作，并应用“一票否决” ---
        df['dynamic_action'] = self._get_dynamic_combat_action()
        euphoria_veto_threshold = get_param_value(p_judge.get('euphoria_veto_threshold'), 0.85)
        df.loc[euphoria_risk_score > euphoria_veto_threshold, 'dynamic_action'] = 'AVOID'

        # --- 步骤 4: 计算最终得分 (应用风险对冲) ---
        df['final_score'] = df['entry_score'] * (1 - attenuation_factor) - df['risk_penalty_score']
        
        # --- 步骤 5: 最终决策 ---
        final_score_threshold = get_param_value(p_judge.get('final_score_threshold'), 400)
        is_score_sufficient = df['final_score'] > final_score_threshold
        not_avoid = df['dynamic_action'] != 'AVOID'
        final_buy_condition = is_score_sufficient & not_avoid
        
        df['signal_type'] = '无信号'
        df.loc[final_buy_condition, 'signal_type'] = '买入信号'
        
        # --- 步骤 5.5: 调用新的风险会计师探针 ---
        debug_params = get_params_block(self.strategy, 'debug_params')
        probe_date_str = get_param_value(debug_params.get('probe_date'))
        if probe_date_str:
            self._deploy_risk_accountant_probe(risk_components_df, probe_date_str)

        # --- 步骤 6: 生成报告与清理 ---
        df['signal_details_cn'] = self._get_human_readable_summary(score_details_df, risk_details_df)
        self._finalize_signals()

    def _deploy_risk_accountant_probe(self, risk_components_df: pd.DataFrame, probe_date: str):
        """
        【V1.0 新增】风险会计师探针
        - 核心职责: 解剖指定日期的 `risk_penalty_score`，清晰列出所有风险项及其贡献分数。
        """
        print("\n" + "="*35 + f" [风险会计师探针 V1.0] 正在审计 {probe_date} 的风险账目 " + "="*35)
        try:
            probe_ts_naive = pd.to_datetime(probe_date)
            if risk_components_df.index.tz is not None:
                probe_ts = probe_ts_naive.tz_localize(risk_components_df.index.tz)
            else:
                probe_ts = probe_ts_naive

            if probe_ts not in risk_components_df.index:
                print(f"  [错误] 探针日期 {probe_date} 不在数据范围内。审计终止。")
                return

            risk_items = risk_components_df.loc[probe_ts]
            # 过滤掉分数为0的项，并按分数从高到低排序
            active_risk_items = risk_items[risk_items > 0].sort_values(ascending=False)

            if active_risk_items.empty:
                print("  [信息] 当日无激活的风险惩罚项。")
            else:
                print("  --- [风险惩罚分构成 (从高到低)] ---")
                for signal_name, score in active_risk_items.items():
                    print(f"    - {signal_name:<60} = {score:.2f}")
            
            total_calculated = active_risk_items.sum()
            final_score_in_df = self.strategy.df_indicators.at[probe_ts, 'risk_penalty_score']
            print("\n  --- [审计总结] ---")
            print(f"  - 探针计算总风险分: {total_calculated:.2f}")
            print(f"  - 最终记录风险分:   {final_score_in_df:.2f}")
            if not np.isclose(total_calculated, final_score_in_df):
                print("  - [审计警告] 探针计算总和与最终记录值不符！请检查计算逻辑。")
            else:
                print("  - [审计通过] 账目核对一致。")

        except Exception as e:
            print(f"  [探针错误] 在执行风险会计师探针时发生异常: {e}")
        finally:
            print("="*95 + "\n")

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
        【V2.7 · 健壮性修复版】生成人类可读的信号摘要。
        - 核心修复 (本次修改):
          - [健壮性] 在构建中文名映射字典(cn_name_map)时，增加了对值类型的检查 (isinstance(v, dict))。
        - 收益: 彻底解决了因信号字典中包含说明性字符串条目（如 "说明_..."）而导致的 AttributeError 崩溃问题。
        """
        # 加载信号与中文名的映射字典
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        
        # 定义所有汇总分数的名称，这些名称不应被剥离前缀
        summary_score_cols = [
            'SCORE_SETUP', 'SCORE_TRIGGER', 'SCORE_PLAYBOOK_SYNERGY',
            'SCORE_REVERSAL_OFFENSE', 'SCORE_RESONANCE_OFFENSE'
        ]
        # 定义需要剥离的前缀列表
        prefixes_to_strip = ['SETUP_', 'DYN_', 'STRATEGIC_', 'BONUS_', 'REVERSAL_', 'RESONANCE_']

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
                # 如果是汇总分、触发器或剧本，直接返回原名，因为它们在字典中的键就是全名
                if (signal_name in summary_score_cols or 
                    signal_name.startswith('TRIGGER_') or 
                    signal_name.startswith('PLAYBOOK_')):
                    return signal_name
                
                # 否则，正常进行前缀剥离
                base_name = signal_name
                for prefix in prefix_list:
                    # 使用 removeprefix 确保只从开头剥离
                    base_name = base_name.removeprefix(prefix)
                return base_name

            # 应用这个智能剥离函数
            long_df['base_signal'] = long_df['signal'].apply(get_base_signal)
            # 向量化映射中文名，只处理值为字典的条目，忽略说明性字符串
            cn_name_map = {k: v.get('cn_name', k) for k, v in score_map.items() if isinstance(v, dict)}
            
            long_df['cn_name'] = long_df['base_signal'].map(cn_name_map).fillna(long_df['base_signal'])
            
            # 调试信息：打印出所有未能成功映射的信号
            unmapped_signals = long_df[long_df['cn_name'] == long_df['base_signal']]['base_signal'].unique()
            if len(unmapped_signals) > 0:
                print(f"    -> [报告层-调试] 以下信号在 signal_dictionary.json 中未找到定义: {list(unmapped_signals)}")

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

    def _calculate_risk_penalty_score(self) -> pd.DataFrame:
        """
        【V400.4 风险会计师版】计算风险惩罚分
        - 核心升级: 不再直接累加到df列，而是构建并返回一个包含所有风险项贡献分的DataFrame。
        """
        atomic = self.strategy.atomic_states
        default_series = pd.Series(0.0, index=self.strategy.df_indicators.index)
        risk_components = {} # 使用字典收集风险项

        def get_clipped_score(signal_name):
            return atomic.get(signal_name, default_series).clip(lower=0)

        # --- 风险1: 绝对否决信号 (权重: 300) ---
        veto_params = get_params_block(self.strategy, 'absolute_veto_params')
        if get_param_value(veto_params.get('enabled'), True):
            mitigation_rules = get_param_value(veto_params.get('mitigation_rules'), {})
            veto_signals = get_param_value(veto_params.get('veto_signals'), [])
            for signal_name in veto_signals:
                risk_score = get_clipped_score(signal_name)
                if risk_score.sum() == 0: continue
                if signal_name in mitigation_rules:
                    mitigators = mitigation_rules[signal_name].get('mitigated_by', [])
                    mitigator_score = pd.Series(0.0, index=self.strategy.df_indicators.index)
                    if mitigators:
                        mitigator_scores = [get_clipped_score(m_signal) for m_signal in mitigators]
                        if mitigator_scores:
                            mitigator_score = pd.Series(np.maximum.reduce([s.values for s in mitigator_scores if not s.empty]), index=self.strategy.df_indicators.index)
                    net_risk_score = risk_score * (1 - mitigator_score)
                    risk_components[signal_name] = net_risk_score * 300
                else:
                    risk_components[signal_name] = risk_score * 300
        
        # --- 风险2: 风险分高于进攻分 (权重: 100) ---
        risk_overrides_score = (self.strategy.df_indicators.get('risk_score', default_series) > self.strategy.df_indicators.get('entry_score', default_series)).astype(float)
        is_in_ascent_phase = get_clipped_score('SCORE_STRUCTURE_MAIN_UPTREND_WAVE_S')
        net_risk_score_override = risk_overrides_score * (1 - is_in_ascent_phase)
        risk_components['RISK_SCORE_GT_OFFENSE'] = net_risk_score_override * 100

        # --- 风险3: 周线战略顶层风险 (权重: 300 for topping, 100 for bearish) ---
        risk_components['CONTEXT_STRATEGIC_TOPPING_RISK_W'] = get_clipped_score('CONTEXT_STRATEGIC_TOPPING_RISK_W') * 300
        risk_components['CONTEXT_STRATEGIC_BEARISH_W'] = get_clipped_score('CONTEXT_STRATEGIC_BEARISH_W') * 100

        # --- 风险4: 顶层认知风险 (权重: 200-300) ---
        risk_components['COGNITIVE_ULTIMATE_BEARISH_CONFIRMATION_S'] = get_clipped_score('COGNITIVE_ULTIMATE_BEARISH_CONFIRMATION_S') * 300
        risk_components['COGNITIVE_SCORE_BREAKDOWN_RESONANCE_S'] = get_clipped_score('COGNITIVE_SCORE_BREAKDOWN_RESONANCE_S') * 300
        risk_components['COGNITIVE_SCORE_TOP_REVERSAL_RESONANCE_S'] = get_clipped_score('COGNITIVE_SCORE_TOP_REVERSAL_RESONANCE_S') * 250
        risk_components['COGNITIVE_SCORE_MULTI_DIMENSIONAL_DIVERGENCE_S'] = get_clipped_score('COGNITIVE_SCORE_MULTI_DIMENSIONAL_DIVERGENCE_S') * 200
        risk_components['COGNITIVE_SCORE_TREND_FATIGUE_RISK'] = get_clipped_score('COGNITIVE_SCORE_TREND_FATIGUE_RISK') * 150

        # --- 风险5: 各情报层S级看跌共振风险 (权重: 150) ---
        risk_components['SCORE_BEHAVIOR_BEARISH_RESONANCE_S'] = get_clipped_score('SCORE_BEHAVIOR_BEARISH_RESONANCE_S') * 150
        risk_components['SCORE_CHIP_BEARISH_RESONANCE_S'] = get_clipped_score('SCORE_CHIP_BEARISH_RESONANCE_S') * 150
        risk_components['SCORE_DYN_BEARISH_RESONANCE_S'] = get_clipped_score('SCORE_DYN_BEARISH_RESONANCE_S') * 150
        risk_components['SCORE_FF_BEARISH_RESONANCE_S'] = get_clipped_score('SCORE_FF_BEARISH_RESONANCE_S') * 150
        risk_components['SCORE_STRUCTURE_BEARISH_RESONANCE_S'] = get_clipped_score('SCORE_STRUCTURE_BEARISH_RESONANCE_S') * 150
        risk_components['SCORE_FOUNDATION_BEARISH_RESONANCE_S'] = get_clipped_score('SCORE_FOUNDATION_BEARISH_RESONANCE_S') * 150

        # --- 风险6: 微观结构风险 (权重: 250-280) ---
        risk_components['COGNITIVE_SCORE_RISK_POWER_SHIFT_TO_RETAIL'] = get_clipped_score('COGNITIVE_SCORE_RISK_POWER_SHIFT_TO_RETAIL') * 280
        risk_components['COGNITIVE_SCORE_RISK_MAIN_FORCE_CONVICTION_WEAKENING'] = get_clipped_score('COGNITIVE_SCORE_RISK_MAIN_FORCE_CONVICTION_WEAKENING') * 250

        # --- 风险7: 亢奋加速风险 (权重: 350) ---
        # 注意：这个风险主要通过衰减因子起作用，这里的惩罚分是二次保险
        risk_components['COGNITIVE_SCORE_RISK_EUPHORIC_ACCELERATION'] = get_clipped_score('COGNITIVE_SCORE_RISK_EUPHORIC_ACCELERATION') * 350

        # --- 风险8: 行业生命周期风险 ---
        industry_params = get_params_block(self.strategy, 'four_layer_scoring_params', {}).get('industry_lifecycle_scoring_params', {})
        if industry_params.get('enabled', False):
            penalty_multiplier = industry_params.get('penalty_multiplier', 600)
            score_stagnation = get_clipped_score('SCORE_INDUSTRY_STAGNATION')
            score_downtrend = get_clipped_score('SCORE_INDUSTRY_DOWNTREND')
            stagnation_weight = industry_params.get('stagnation_penalty_weight', 1.2)
            downtrend_weight = industry_params.get('downtrend_penalty_weight', 1.5)
            risk_components['RISK_INDUSTRY_STAGNATION'] = score_stagnation * stagnation_weight * penalty_multiplier
            risk_components['RISK_INDUSTRY_DOWNTREND'] = score_downtrend * downtrend_weight * penalty_multiplier
        
        return pd.DataFrame(risk_components).fillna(0)

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














