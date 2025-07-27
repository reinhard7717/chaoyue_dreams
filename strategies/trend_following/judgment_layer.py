# 文件: strategies/trend_following/judgment_layer.py
# 统合判断层
import pandas as pd
from .utils import get_params_block, get_param_value
from .intelligence_layer import MainForceState

class JudgmentLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def make_final_decisions(self):
        print("    --- [最高作战指挥部 V304.0] 启动，正在执行“评估-决策”一体化流程... ---")
        df = self.strategy.df_indicators
        
        df['final_score'] = 0.0
        df['signal_type'] = '中性'
        df['signal_entry'] = False
        df['exit_signal_code'] = 0
        df['exit_severity_level'] = 0

        is_potential_buy = df['entry_score'] > 0
        risk_overrides_entry = df['risk_score'] > df['entry_score']
        df.loc[is_potential_buy, 'signal_type'] = '买入信号'
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
