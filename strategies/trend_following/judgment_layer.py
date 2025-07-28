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
        【V317.0 动态力学版】
        - 核心革命: 引入“动态力学战术矩阵”，基于进攻与风险的“双向加速度”进行最终裁决。
        - 决策流程:
          1. 联席会议投票，形成基础否决票数。
          2. 启动战术矩阵，根据四个“力学象限”给出最终的“行动指令”（强攻/暂缓/规避/死守）。
          3. 结合“黄金买点”等特殊指令，形成最终决策。
        """
        print("    --- [最高作战指挥部 V317.0 动态力学版] 启动... ---")
        df = self.strategy.df_indicators
        
        df['final_score'], df['signal_type'], df['signal_entry'] = 0.0, '中性', False
        df['exit_signal_code'], df['exit_severity_level'], df['veto_votes'] = 0, 0, 0

        is_potential_buy = df['entry_score'] > 0
        
        # 步骤1: 联席会议投票，计算基础否决票数
        self._calculate_static_veto_votes()

        # 步骤2: 启动“动态力学战术矩阵”进行最终裁决
        print("        -> [战术矩阵] 正在启动“动态力学战术矩阵”...")
        final_action = self._get_dynamic_combat_action()
        
        # 步骤3: 结合静态评估和动态指令，形成最终买入条件
        # 基础条件：有买入意图，且静态否决票在可容忍范围内
        base_buy_condition = is_potential_buy & (
            (df['entry_score'] < 800) & (df['veto_votes'] <= 1) |
            (df['entry_score'] >= 800) & (df['veto_votes'] <= 3)
        )
        
        # 最终条件：满足基础条件，并且动态指令不是“规避”
        tactical_buy_condition = base_buy_condition & (final_action != 'AVOID')
        
        # 动态指令可以覆盖基础条件：如果指令是“强攻”，则无视部分否决票
        force_attack_condition = is_potential_buy & (final_action == 'FORCE_ATTACK')
        
        final_buy_condition = tactical_buy_condition | force_attack_condition

        # 特殊情况：黄金买点，拥有最高优先权，但不能处于“死亡象限”
        prev_state = df['main_force_state'].shift(1)
        is_entering_markup = (prev_state.isin([MainForceState.ACCUMULATING.value, MainForceState.WASHING.value]) & (df['main_force_state'] == MainForceState.MARKUP.value))
        golden_buy_point = is_entering_markup & (final_action != 'AVOID')
        
        final_buy_condition |= golden_buy_point

        # 应用最终决策
        df.loc[final_buy_condition, 'signal_type'] = '买入信号'
        df.loc[is_potential_buy & ~final_buy_condition, 'signal_type'] = '卖出信号'

        # --- 后续处理 ---
        self.strategy.exit_layer.calculate_exit_signals()
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
        """计算基础的、静态的否决票数"""
        print("        -> [联席会议] 正在进行静态否决票评估...")
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)

        # 筹码地基风险 (3票)
        has_critical_chip_risk = atomic.get('RISK_CHIP_STRUCTURE_CRITICAL_FAILURE', default_series)
        df.loc[has_critical_chip_risk, 'veto_votes'] += 3

        # 主力行为风险 (1票)
        is_in_distribution_phase = df['main_force_state'].isin([MainForceState.DISTRIBUTING.value, MainForceState.COLLAPSE.value])
        df.loc[is_in_distribution_phase, 'veto_votes'] += 1
        
        # 其他常规风险 (每项1票)
        risk_overrides_entry = df['risk_score'] > df['entry_score']
        is_in_ascent_phase = atomic.get('STRUCTURE_POST_ACCUMULATION_ASCENT_C', default_series)
        df.loc[risk_overrides_entry & ~is_in_ascent_phase, 'veto_votes'] += 1

    def _finalize_signals(self):
        """整理并最终确定信号列"""
        df = self.strategy.df_indicators
        final_buy_condition = df['signal_type'] == '买入信号'
        df.loc[final_buy_condition, 'final_score'] = df.loc[final_buy_condition, 'entry_score']
        df.loc[final_buy_condition, 'signal_entry'] = True
        df.loc[final_buy_condition, ['exit_signal_code', 'exit_severity_level']] = 0

        final_sell_condition = (df['signal_type'] == '卖出信号')
        df.loc[final_sell_condition, 'final_score'] = df.loc[final_sell_condition, 'entry_score']
        df.loc[final_sell_condition, 'signal_entry'] = False
        
        is_explicit_exit = df['exit_signal_code'] >= 88
        df.loc[is_explicit_exit, 'signal_type'] = '卖出信号'

        print("        -> [决策单元] 决策完成。正在进行最终分数审查...")
        final_check_df = df[(df['signal_type'] != '中性')].tail(5)
        if not final_check_df.empty:
            print("          -> [最终分数审查报告]:")
            print(final_check_df[['entry_score', 'risk_score', 'veto_votes', 'final_score', 'signal_type', 'main_force_state']])
        else:
            print("          -> [最终分数审查报告]: 未发现任何有效信号。")

