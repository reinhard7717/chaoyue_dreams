# 文件: strategies/trend_following/judgment_layer.py
# 统合判断层
import pandas as pd
import numpy as np # 确保numpy也被导入
from scipy.stats import linregress # 导入线性回归函数
from .utils import get_params_block, get_param_value
from .intelligence_layer import MainForceState

class JudgmentLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def make_final_decisions(self):
        """
        【V316.0 筹码加权版】
        - 核心重构: 恢复并强化“联席会议”加权否决票体系。
        - 筹码地基: “严重筹码结构风险”将投出3张否决票，权重最高。
        - 动态容忍: 进攻分越高的信号，能容忍的否决票越多，为极端机会保留可能。
        - 决策流程: 筹码风险 -> 主力行为 -> 其他风险 -> 联席会议投票 -> 最终裁决。
        """
        print("    --- [最高作战指挥部 V316.0 筹码加权版] 启动... ---")
        df = self.strategy.df_indicators
        
        df['final_score'] = 0.0
        df['signal_type'] = '中性'
        df['signal_entry'] = False
        df['exit_signal_code'] = 0
        df['exit_severity_level'] = 0
        df['veto_votes'] = 0

        self._diagnose_score_dynamics()
        
        # --- 联席会议开始：各部门提交否决意见 ---
        is_potential_buy = df['entry_score'] > 0
        default_series = pd.Series(False, index=df.index)

        # 1. 【最高权重】筹码地基审查
        print("        -> [联席会议] 正在执行“筹码地基”审查...")
        has_critical_chip_risk = self.strategy.atomic_states.get('RISK_CHIP_STRUCTURE_CRITICAL_FAILURE', default_series)
        df.loc[has_critical_chip_risk, 'veto_votes'] += 3 # 投出3票！
        if has_critical_chip_risk.any():
            print(f"          -> [筹码地基审查] 发现严重筹码结构风险，在 {has_critical_chip_risk.sum()} 天内投出3张否决票！")

        # 2. 主力行为分析部
        print("        -> [联席会议] 正在执行“主力行为”审查...")
        is_in_distribution_phase = df['main_force_state'].isin([MainForceState.DISTRIBUTING.value, MainForceState.COLLAPSE.value])
        df.loc[is_in_distribution_phase, 'veto_votes'] += 1
        if is_in_distribution_phase.any():
            print(f"          -> [主力行为审查] 因处于“派发/崩盘期”，在 {is_in_distribution_phase.sum()} 天内投出1张否决票。")

        # 3. 军事监察部：评估“绝对否决权”风险（非筹码类）
        print("        -> [联席会议] 正在执行“绝对否决权”审查...")
        veto_params = get_params_block(self.strategy, 'absolute_veto_params')
        if get_param_value(veto_params.get('enabled'), True):
            # 注意：这里的 veto_signals 应该排除已在筹码审查中处理过的信号
            chip_risks_in_veto = {
                "RISK_CAPITAL_STRUCT_MAIN_FORCE_DISTRIBUTING", # 假设这个与主力行为有关
                "CONTEXT_RECENT_DISTRIBUTION_PRESSURE" # 假设这个与主力行为有关
            }
            veto_signals = [s for s in get_param_value(veto_params.get('veto_signals'), []) if s not in chip_risks_in_veto]
            mitigation_rules = get_param_value(veto_params.get('mitigation_rules'), {})
            
            final_absolute_veto = pd.Series(False, index=df.index)
            for signal_name in veto_signals:
                has_risk = self.strategy.atomic_states.get(signal_name, default_series)
                if signal_name in mitigation_rules:
                    mitigators = mitigation_rules[signal_name].get('mitigated_by', [])
                    has_mitigator = pd.Series(False, index=df.index)
                    for m_signal in mitigators: has_mitigator |= self.strategy.atomic_states.get(m_signal, default_series)
                    final_absolute_veto |= (has_risk & ~has_mitigator)
                else:
                    final_absolute_veto |= has_risk
            df.loc[final_absolute_veto, 'veto_votes'] += 2 # 投出2票
            if final_absolute_veto.any():
                print(f"          -> [绝对否决权审查] 在 {final_absolute_veto.sum()} 天内投出2张否决票。")

        # 4. 其他常规风险评估
        print("        -> [联席会议] 正在执行其他常规风险评估...")
        is_in_ascent_phase = self.strategy.atomic_states.get('STRUCTURE_POST_ACCUMULATION_ASCENT_C', default_series)
        risk_overrides_entry = df['risk_score'] > df['entry_score']
        regular_risk_veto = risk_overrides_entry & ~is_in_ascent_phase
        df.loc[regular_risk_veto, 'veto_votes'] += 1

        is_opportunity_fading = self.strategy.atomic_states.get('SCORE_DYN_OPPORTUNITY_FADING', default_series)
        df.loc[is_opportunity_fading, 'veto_votes'] += 1
        is_risk_escalating = self.strategy.atomic_states.get('SCORE_DYN_RISK_ESCALATING', default_series)
        df.loc[is_risk_escalating, 'veto_votes'] += 1

        # --- 最高指挥部最终裁决：基于进攻分和总否决票数 ---
        print("        -> [最高指挥部] 正在根据“进攻分”与“总否决票数”进行最终裁决...")
        
        # 动态容忍度规则
        buy_condition = (
            is_potential_buy & 
            (
                (df['entry_score'] < 400) & (df['veto_votes'] == 0) |
                (df['entry_score'].between(400, 799)) & (df['veto_votes'] <= 1) |
                (df['entry_score'].between(800, 1199)) & (df['veto_votes'] <= 2) |
                (df['entry_score'] >= 1200) & (df['veto_votes'] <= 3) # 只有天选之子才能承受筹码地基风险
            )
        )

        # 特殊情况：黄金买点，拥有最高优先权，但仍需通过筹码地基审查
        prev_state = df['main_force_state'].shift(1)
        is_entering_markup = (prev_state.isin([MainForceState.ACCUMULATING.value, MainForceState.WASHING.value]) & (df['main_force_state'] == MainForceState.MARKUP.value))
        golden_buy_point = is_entering_markup & ~has_critical_chip_risk
        if golden_buy_point.any():
            print(f"          -> [最高指令] “黄金买点”已确认！在 {golden_buy_point.sum()} 天内，无视战术否决票，强制生成买入信号。")
        
        final_buy_condition = buy_condition | golden_buy_point
        
        df.loc[final_buy_condition, 'signal_type'] = '买入信号'
        df.loc[is_potential_buy & ~final_buy_condition, 'signal_type'] = '卖出信号'

        # --- 后续处理 ---
        self.strategy.exit_layer.calculate_exit_signals()
        
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

    def _diagnose_score_dynamics(self):
        """
        【V311.1 线性回归升级版】元情报诊断模块
        - 核心升级: 使用基于5日滚动窗口的线性回归计算斜率，取代简单的差分。
                    这能有效过滤掉分数的日常噪音，捕捉更可靠的趋势变化，
                    极大提升了“元状态”的信号质量。
        """
        print("        -> [元情报诊断单元 V311.1] 正在分析分数动态(线性回归)...")
        df = self.strategy.df_indicators
        
        # 定义滚动窗口
        window = 5

        # 使用 apply 和线性回归计算滚动斜率
        # np.arange(window) 创建一个 [0, 1, 2, 3, 4] 的时间序列作为 x
        # .iloc[-1] 是因为 linregress 返回多个值，我们只需要斜率(slope)
        entry_score_slope = df['entry_score'].rolling(window).apply(
            lambda y: linregress(np.arange(window), y).slope, raw=False
        )
        risk_score_slope = df['risk_score'].rolling(window).apply(
            lambda y: linregress(np.arange(window), y).slope, raw=False
        )

        # 定义斜率阈值，防止趋势过于平缓时产生信号
        # 对于线性回归斜率，阈值可以设置得更小
        opportunity_threshold = 2.0 
        risk_threshold = 2.0

        # 生成元状态并存入 atomic_states
        self.strategy.atomic_states['SCORE_DYN_OPPORTUNITY_RISING'] = entry_score_slope > opportunity_threshold
        self.strategy.atomic_states['SCORE_DYN_OPPORTUNITY_FADING'] = entry_score_slope < -opportunity_threshold
        self.strategy.atomic_states['SCORE_DYN_RISK_ESCALATING'] = risk_score_slope > risk_threshold
        self.strategy.atomic_states['SCORE_DYN_RISK_SUBSIDING'] = risk_score_slope < -risk_threshold
        
        print("          -> [元情报] “机会/风险”的动态趋势已生成。")









