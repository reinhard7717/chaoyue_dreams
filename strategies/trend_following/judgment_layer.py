# 文件: strategies/trend_following/judgment_layer.py
# 统合判断层 (V317.0 - 动态力学版)
import pandas as pd
import numpy as np
from .intelligence_layer import MainForceState
from .utils import get_params_block, get_param_value

class JudgmentLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def make_final_decisions(self):
        """
        【V318.3 终极修复版】
        - 核心修复: 纠正了对 `exit_layer.calculate_exit_signals()` 的调用路径，
                    确保通过 `self.strategy.exit_layer` 进行正确调用。
        - 逻辑强化: 继承并确认了“风控回归”和“主动净化”逻辑，确保决策的
                    稳健性和报告的纯净性。
        """
        print("    --- [最高作战指挥部 V318.3 终极修复版] 启动... ---")
        df = self.strategy.df_indicators
        
        df['final_score'], df['signal_type'], df['signal_entry'] = 0.0, '中性', False
        df['exit_signal_code'], df['exit_severity_level'], df['veto_votes'] = 0, 0, 0
        df['dynamic_action'] = 'HOLD'

        is_potential_buy = df['entry_score'] > 0
        
        # 步骤1: 主动预防性净化
        print("        -> [决策预处理] 正在对所有潜在买入日执行“主动净化”...")
        df.loc[is_potential_buy, ['exit_signal_code', 'exit_severity_level', 'alert_reason']] = [0, 0, '']

        # 步骤2: 联席会议投票
        self._calculate_static_veto_votes()

        # 步骤3: 动态力学矩阵裁决
        print("        -> [战术矩阵] 正在启动“动态力学战术矩阵”...")
        final_action = self._get_dynamic_combat_action()
        df['dynamic_action'] = final_action
        
        # 步骤4: 形成最终买入条件
        base_buy_condition = is_potential_buy & (
            (df['entry_score'] < 800) & (df['veto_votes'] <= 1) |
            (df['entry_score'].between(800, 1199)) & (df['veto_votes'] <= 2) |
            (df['entry_score'] >= 1200) & (df['veto_votes'] <= 3)
        )
        tactical_buy_condition = base_buy_condition & (final_action != 'AVOID')
        force_attack_condition = is_potential_buy & (final_action == 'FORCE_ATTACK') & (df['veto_votes'] <= 3)
        final_buy_condition = tactical_buy_condition | force_attack_condition

        prev_state = df['main_force_state'].shift(1)
        is_entering_markup = (prev_state.isin([MainForceState.ACCUMULATING.value, MainForceState.WASHING.value]) & (df['main_force_state'] == MainForceState.MARKUP.value))
        golden_buy_point = is_entering_markup & (final_action != 'AVOID')
        final_buy_condition |= golden_buy_point

        # 应用初步决策
        df.loc[final_buy_condition, 'signal_type'] = '买入信号'
        df.loc[is_potential_buy & ~final_buy_condition, 'signal_type'] = '卖出信号'

        # --- 步骤5: 后续处理 ---
        # 【核心修复】: 使用正确的路径 `self.strategy.exit_layer` 调用
        self.strategy.exit_layer.calculate_exit_signals()
        
        # 最终信号确定与净化
        self._finalize_signals()

    def _get_dynamic_combat_action(self) -> pd.Series:
        """
        【V317.0 核心】动态力学战术矩阵
        根据进攻和风险的“双向加速度”状态，返回四种战术指令之一：
        - FORCE_ATTACK (强攻): 进攻加速，风险减速。这是最佳战机。
        - PROCEED_WITH_CAUTION (暂缓/谨慎前行): 双向加速或双向减速。
        - AVOID (规避): 进攻减速，风险加速。这是最危险的陷阱。
        - HOLD (死守): 默认状态。
        """
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)
        
        offense_accel = atomic.get('FORCE_VECTOR_OFFENSE_ACCELERATING', default_series)
        offense_decel = atomic.get('FORCE_VECTOR_OFFENSE_DECELERATING', default_series)
        risk_accel = atomic.get('FORCE_VECTOR_RISK_ACCELERATING', default_series)
        risk_decel = atomic.get('FORCE_VECTOR_RISK_DECELERATING', default_series)

        # 1. 黄金象限 (Golden Quadrant): 进攻加速，风险减速
        is_force_attack = offense_accel & risk_decel
        
        # 2. 死亡象限 (Death Quadrant): 进攻减速，风险加速
        is_avoid = offense_decel & risk_accel
        
        # 3. 缠斗象限 (Contested Quadrant): 其他情况
        is_caution = (offense_accel & risk_accel) | (offense_decel & risk_decel)

        # 生成最终行动指令序列
        actions = pd.Series('HOLD', index=df.index)
        actions.loc[is_force_attack] = 'FORCE_ATTACK'
        actions.loc[is_avoid] = 'AVOID'
        actions.loc[is_caution] = 'PROCEED_WITH_CAUTION'
        
        if is_force_attack.any(): print(f"          -> [战术矩阵] 在 {is_force_attack.sum()} 天内发出“强攻”指令！")
        if is_avoid.any(): print(f"          -> [战术矩阵] 在 {is_avoid.sum()} 天内发出“规避”指令！")
        
        return actions

    def _calculate_static_veto_votes(self):
        """
        【V318.1 风控回归版】
        - 核心修复: 将 `risk_score > entry_score` 这个基础风控重新纳入否决票体系。
                    它现在作为常规风险，与其他风险因素一起参与投票。
        """
        print("        -> [联席会议] 正在进行静态否决票评估...")
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)

        # 1. 筹码地基风险 (3票)
        has_critical_chip_risk = atomic.get('RISK_CHIP_STRUCTURE_CRITICAL_FAILURE', default_series)
        df.loc[has_critical_chip_risk, 'veto_votes'] += 3

        # 2. 主力行为风险 (1票)
        is_in_distribution_phase = df['main_force_state'].isin([MainForceState.DISTRIBUTING.value, MainForceState.COLLAPSE.value])
        df.loc[is_in_distribution_phase, 'veto_votes'] += 1
        
        # 3. 绝对否决权风险 (2票)
        veto_params = get_params_block(self.strategy, 'absolute_veto_params')
        if get_param_value(veto_params.get('enabled'), True):
            mitigation_rules = get_param_value(veto_params.get('mitigation_rules'), {})
            chip_risks_in_veto = {"RISK_CAPITAL_STRUCT_MAIN_FORCE_DISTRIBUTING", "CONTEXT_RECENT_DISTRIBUTION_PRESSURE"}
            veto_signals = [s for s in get_param_value(veto_params.get('veto_signals'), []) if s not in chip_risks_in_veto]
            
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

        # 4. 【核心修复】常规风险 (每项1票)
        # 4.1 风险/收益倒挂
        risk_overrides_entry = df['risk_score'] > df['entry_score']
        is_in_ascent_phase = atomic.get('STRUCTURE_POST_ACCUMULATION_ASCENT_C', default_series)
        df.loc[risk_overrides_entry & ~is_in_ascent_phase, 'veto_votes'] += 1
        
        # 4.2 机会衰退
        is_opportunity_fading = atomic.get('SCORE_DYN_OPPORTUNITY_FADING', default_series)
        df.loc[is_opportunity_fading, 'veto_votes'] += 1
        
        # 4.3 风险抬头
        is_risk_escalating = atomic.get('SCORE_DYN_RISK_ESCALATING', default_series)
        df.loc[is_risk_escalating, 'veto_votes'] += 1

    def _finalize_signals(self):
        """
        【V318.2 终极净化版】
        - 核心修复: 将信号净化步骤移至整个决策流程的绝对末端。
        - 新逻辑: 在确定了最终的买入/卖出信号后，再进行净化。
                  - 对最终的买入信号，强制清除所有与之冲突的卖出/预警信息。
                  - 对最终的卖出信号，保留其卖出信息。
        - 收益: 彻底根除因计算顺序导致的“决策幽灵”问题，确保报告的绝对纯净。
        """
        df = self.strategy.df_indicators
        
        # 1. 识别最终的信号类型
        final_buy_condition = df['signal_type'] == '买入信号'
        final_sell_condition = df['signal_type'] == '卖出信号'

        # 2. 为买入信号设置最终状态并执行“终极净化”
        df.loc[final_buy_condition, 'final_score'] = df.loc[final_buy_condition, 'entry_score']
        df.loc[final_buy_condition, 'signal_entry'] = True
        df.loc[final_buy_condition, ['exit_signal_code', 'exit_severity_level', 'alert_reason']] = [0, 0, '']

        # 3. 为卖出信号设置最终状态
        df.loc[final_sell_condition, 'final_score'] = df.loc[final_sell_condition, 'entry_score']
        df.loc[final_sell_condition, 'signal_entry'] = False
        
        # 4. 确保由离场层生成的明确卖出代码被正确标记
        is_explicit_exit = df['exit_signal_code'] >= 88
        df.loc[is_explicit_exit, 'signal_type'] = '卖出信号'

        print("        -> [决策单元] 决策完成。正在进行最终分数审查...")
        final_check_df = df[(df['signal_type'] != '中性')].tail(5)
        if not final_check_df.empty:
            print("          -> [最终分数审查报告]:")
            print(final_check_df[['entry_score', 'risk_score', 'veto_votes', 'final_score', 'signal_type', 'main_force_state']])
        else:
            print("          -> [最终分数审查报告]: 未发现任何有效信号。")

