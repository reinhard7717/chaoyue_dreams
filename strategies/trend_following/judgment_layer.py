# 文件: strategies/trend_following/judgment_layer.py
# 统合判断层 (V317.0 - 动态力学版)
import pandas as pd
import numpy as np
from .intelligence_layer import MainForceState
from .utils import get_params_block, get_param_value

class JudgmentLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def _evaluate_holding_health(self):
        """
        【新增】持仓大脑 - 评估持仓健康度 (HHS)
        - 核心职责: 在持仓期间，每日计算一个“持仓健康分”，
                    并根据分数变化定义不同的持仓状态。
        """
        print("        -> [持仓大脑] 正在评估所有活跃持仓的健康状况...")
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)

        # --- 1. 定义健康评分的构成项 (正向/负向) ---
        health_factors = {
            # 趋势强度 (最高权重)
            'FRACTAL_STATE_STRONG_TREND': 30,
            'DYN_TREND_HEALTHY_ACCELERATING': 25,
            'STRUCTURE_MAIN_UPTREND_WAVE_S': 20,
            # 主力行为
            'COGNITIVE_PATTERN_LOCK_CHIP_RALLY': 30,
            'MECHANICS_COST_ACCELERATING': 20,
            'CPA_RISE_WITH_MAIN_FORCE_SUPPORT': 15,
            'MECHANICS_INERTIA_DECREASING': 15,
        }
        
        worsening_factors = {
            # 结构性风险 (最高权重)
            'FRACTAL_RISK_TOP_DIVERGENCE': -50,
            'FRACTAL_EVENT_TREND_EXHAUSTION': -40,
            'STRUCTURE_TOPPING_DANGER_S': -40,
            # 主力派发
            'RISK_CAPITAL_STRUCT_MAIN_FORCE_DISTRIBUTING': -30,
            'CONTEXT_RECENT_DISTRIBUTION_PRESSURE': -20,
            # 趋势减弱
            'DYN_TREND_WEAKENING_DECELERATING': -20,
            'RISK_DYN_DIVERGING': -15,
        }

        # --- 2. 计算每日的健康分 ---
        hhs = pd.Series(0.0, index=df.index)
        for factor, score in health_factors.items():
            hhs += atomic.get(factor, default_series) * score
        
        for factor, score in worsening_factors.items():
            hhs += atomic.get(factor, default_series) * score
            
        df['holding_health_score'] = hhs

        # --- 3. 定义持仓状态 (状态机) ---
        # 状态1: 强力持仓 (可以考虑加仓)
        df['HOLDING_STATE_STRONG'] = (hhs >= 50) & (df['risk_score'] < 300)
        
        # 状态2: 正常持仓 (继续持有)
        df['HOLDING_STATE_NORMAL'] = (hhs.between(20, 49)) & (df['risk_score'] < 500)
        
        # 状态3: 警告状态 (应上移止损，准备减仓)
        df['HOLDING_STATE_WARNING'] = (hhs < 20) | (df['risk_score'] >= 500)
        
        # 状态4: 风险恶化 (应主动减仓)
        # 例如：健康分连续3天下降，且今日为负
        is_hhs_declining_3d = (hhs < hhs.shift(1)) & (hhs.shift(1) < hhs.shift(2))
        df['HOLDING_STATE_DETERIORATING'] = is_hhs_declining_3d & (hhs < 0)

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

        # --- 【新增】步骤0: 评估持仓健康度 ---
        # 这个评估独立于买入决策，为后续的持仓管理提供依据
        self._evaluate_holding_health()
        
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
        self.strategy.exit_layer.calculate_exit_signals()

        # --- 步骤6: 根据持仓状态，生成“持仓建议”信号 ---
        # 这个信号不直接触发买卖，而是作为一种“预警”或“建议”
        is_strong_hold = df.get('HOLDING_STATE_STRONG', pd.Series(False, index=df.index))
        is_warning_hold = df.get('HOLDING_STATE_WARNING', pd.Series(False, index=df.index))
        is_deteriorating = df.get('HOLDING_STATE_DETERIORATING', pd.Series(False, index=df.index))

        # 我们可以在 alert_reason 中增加这些状态信息
        # 注意：这里的 .loc 条件要确保不覆盖已有的卖出信号
        no_exit_signal = df['exit_signal_code'] == 0
        df.loc[is_strong_hold & no_exit_signal, 'alert_reason'] = '强力持仓(可加仓)'
        df.loc[is_warning_hold & no_exit_signal, 'alert_reason'] = '持仓预警(收紧止损)'
        df.loc[is_deteriorating & no_exit_signal, 'alert_reason'] = '健康度恶化(应减仓)'
        
        # 对于“健康度恶化”，我们可以生成一个1级预警
        df.loc[is_deteriorating & no_exit_signal, 'alert_level'] = 10 # 使用一个自定义的level
        df.loc[is_deteriorating & no_exit_signal, 'is_risk_warning'] = True

        # --- 步骤7: 最终信号确定与净化 ---
        self._finalize_signals()
        
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

