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
        print("    --- [最高作战指挥部 V304.0] 启动，正在执行“评估-决策”一体化流程... ---")
        df = self.strategy.df_indicators
        
        self._diagnose_score_dynamics()
        
        df['final_score'] = 0.0
        df['signal_type'] = '中性'
        df['signal_entry'] = False
        df['exit_signal_code'] = 0
        df['exit_severity_level'] = 0

        is_potential_buy = df['entry_score'] > 0
        risk_overrides_entry = df['risk_score'] > df['entry_score']
        df.loc[is_potential_buy, 'signal_type'] = '买入信号'

        print("        -> [战略参谋部] 正在评估“初升浪”战术豁免权...")
        is_in_ascent_phase = self.strategy.atomic_states.get('STRUCTURE_POST_ACCUMULATION_ASCENT_C', pd.Series(False, index=df.index))
        
        risk_overrides_entry = df['risk_score'] > df['entry_score']
        # 核心修改：只有在“非初升浪”期间，风险分才能常规地否决进攻分
        final_risk_override_condition = is_potential_buy & risk_overrides_entry & ~is_in_ascent_phase
        
        df.loc[final_risk_override_condition, 'signal_type'] = '卖出信号'
        
        # 打印豁免日志
        exempted_days = is_potential_buy & risk_overrides_entry & is_in_ascent_phase
        if exempted_days.any():
             print(f"          -> [豁免报告] “初升浪”豁免权已触发！在 {exempted_days.sum()} 天内，风险分被禁止否决进攻分，以保护趋势。")
        
        df.loc[is_potential_buy & risk_overrides_entry, 'signal_type'] = '卖出信号'

        print("        -> [军事监察部] 正在执行“绝对否决权”审查...")
        veto_params = get_params_block(self.strategy, 'absolute_veto_params')
        if get_param_value(veto_params.get('enabled'), True):
            veto_signals = get_param_value(veto_params.get('veto_signals'), [])
            has_absolute_veto_risk = pd.Series(False, index=df.index)
            for signal_name in veto_signals:
                risk_series = self.strategy.atomic_states.get(signal_name, pd.Series(False, index=df.index))
                has_absolute_veto_risk |= risk_series
            if has_absolute_veto_risk.any():
                df.loc[has_absolute_veto_risk, 'signal_type'] = '卖出信号'
                print(f"          -> [审查报告] “绝对否决权”已触发！在 {has_absolute_veto_risk.sum()} 天内强制覆盖了买入信号。")

        print("        -> [战略参谋部] 正在执行“主力行为序列”审查...")
        is_in_distribution_phase = df['main_force_state'].isin([MainForceState.DISTRIBUTING.value, MainForceState.COLLAPSE.value])
        df.loc[is_in_distribution_phase, 'signal_type'] = '卖出信号'
        if is_in_distribution_phase.any():
            print(f"          -> [战略报告] “派发/崩盘期”否决已触发！在 {is_in_distribution_phase.sum()} 天内强制覆盖了买入信号。")

        prev_state = df['main_force_state'].shift(1)
        is_entering_markup = (prev_state.isin([MainForceState.ACCUMULATING.value, MainForceState.WASHING.value]) & (df['main_force_state'] == MainForceState.MARKUP.value))
        df.loc[is_entering_markup, 'signal_type'] = '买入信号'
        if is_entering_markup.any():
            print(f"          -> [战略报告] “黄金买点”已确认！在 {is_entering_markup.sum()} 天内发现了战略建仓机会。")

        print("        -> [元决策单元] 正在应用分数动态进行最终裁决...")
        # 裁决1: 如果机会正在衰退，则否决买入信号 (防范冲高回落的陷阱)
        is_opportunity_fading = self.strategy.atomic_states.get('SCORE_DYN_OPPORTUNITY_FADING', pd.Series(False, index=df.index))
        df.loc[is_opportunity_fading, 'signal_type'] = '卖出信号'
        if (is_opportunity_fading & is_potential_buy).any():
            print(f"          -> [元裁决] “机会衰退”否决已触发！在 {(is_opportunity_fading & is_potential_buy).sum()} 天内否决了潜在买点。")

        # 裁决2: 如果风险正在抬头，则否决买入信号 (防范风险累积)
        is_risk_escalating = self.strategy.atomic_states.get('SCORE_DYN_RISK_ESCALATING', pd.Series(False, index=df.index))
        df.loc[is_risk_escalating, 'signal_type'] = '卖出信号'
        if (is_risk_escalating & is_potential_buy).any():
            print(f"          -> [元裁决] “风险抬头”否决已触发！在 {(is_risk_escalating & is_potential_buy).sum()} 天内否决了潜在买点。")

        # 调用离场层
        self.strategy.exit_layer.calculate_exit_signals()

        is_pure_sell = df['exit_signal_code'] >= 88
        df.loc[is_pure_sell, 'signal_type'] = '卖出信号'

        final_buy_condition = df['signal_type'] == '买入信号'
        df.loc[final_buy_condition, 'final_score'] = df.loc[final_buy_condition, 'entry_score']
        df.loc[final_buy_condition, 'signal_entry'] = True
        df.loc[final_buy_condition, ['exit_signal_code', 'exit_severity_level']] = 0

        print("        -> [决策单元] 决策完成。正在进行最终分数审查...")
        final_check_df = df[(df['signal_type'] != '中性')].tail(5)
        if not final_check_df.empty:
            print("          -> [最终分数审查报告]:")
            print(final_check_df[['entry_score', 'risk_score', 'final_score', 'signal_type', 'exit_severity_level', 'main_force_state']])
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









