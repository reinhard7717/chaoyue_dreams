# 文件: strategies/trend_following/performance_analyzer.py
# 升级模块：性能分析器 (V3.0 深度复盘版)
import pandas as pd
import numpy as np
from .utils import get_param_value

class PerformanceAnalyzer:
    """
    【V3.0 深度复盘版】策略性能分析器
    - 核心升级: 模拟交易不再只返回简单的“成功/失败”，而是返回一个包含
                最大收益/回撤、所需天数、退出原因等详细信息的字典。
    - 聚合能力: 聚合报告现在能计算胜率、平均盈亏、平均持有时长等更丰富的指标。
    """
    def __init__(self, df_indicators: pd.DataFrame, score_details_df: pd.DataFrame, analysis_params: dict, scoring_params: dict):
        """
        初始化分析器
        :param df_indicators: 包含最终信号和K线数据的主DataFrame。
        :param score_details_df: 包含每日各信号得分详情的DataFrame。
        :param analysis_params: 性能分析模块的专属配置。
        :param scoring_params: 四层计分模型的配置，用于获取信号元数据。
        """
        self.df = df_indicators
        self.score_details_df = score_details_df
        self.analysis_params = analysis_params
        self.scoring_params = scoring_params
        if self.df is None or self.df.empty:
            raise ValueError("PerformanceAnalyzer 接收到的 df_indicators 为空。")
        if self.score_details_df is None or self.score_details_df.empty:
            raise ValueError("PerformanceAnalyzer 接收到的 score_details_df 为空。")
        
        # 从配置中获取分析参数
        self.look_forward_days = get_param_value(self.analysis_params.get('look_forward_days'), 20)
        self.profit_target_pct = get_param_value(self.analysis_params.get('profit_target_pct'), 0.15)
        self.stop_loss_pct = get_param_value(self.analysis_params.get('stop_loss_pct'), 0.07)

    def run_analysis(self) -> list:
        """
        运行性能分析的主函数，并返回结构化的结果列表。
        """
        buy_signals = self.df[self.df['signal_type'] == '买入信号']
        if buy_signals.empty:
            return []
            
        trade_outcomes = []
        for entry_date, row in buy_signals.iterrows():
            # 【代码修改】调用新的、功能更强大的分析方法
            outcome_details = self._analyze_single_trade_performance(entry_date)
            if outcome_details:
                trade_outcomes.append({
                    'entry_date': entry_date,
                    **outcome_details  # 将详细结果字典解包合并
                })
        
        # 【代码修改】调用新的聚合报告方法
        return self._aggregate_and_report_v2(trade_outcomes)

    def _analyze_single_trade_performance(self, entry_date) -> dict:
        """
        【V3.0 新增核心方法】深度分析单次交易的性能表现。
        - 返回一个包含详细指标的字典，而不仅仅是成功/失败。
        """
        try:
            entry_price = self.df.loc[entry_date, 'close_D']
            entry_idx = self.df.index.get_loc(entry_date)
            # 安全地切片，防止越界
            look_forward_df = self.df.iloc[entry_idx + 1 : entry_idx + 1 + self.look_forward_days]
        except (KeyError, IndexError):
            return None # 如果找不到日期或索引，返回None

        if look_forward_df.empty:
            return None

        target_price = entry_price * (1 + self.profit_target_pct)
        stop_price = entry_price * (1 - self.stop_loss_pct)

        # 初始化性能指标
        max_profit_pct = 0.0
        days_to_max_profit = 0
        max_drawdown_pct = 0.0
        days_to_max_drawdown = 0
        exit_reason = 'timeout'
        exit_days = self.look_forward_days
        final_outcome = 'timeout'

        # 逐日分析未来走势
        for i, (date, row) in enumerate(look_forward_df.iterrows()):
            day_num = i + 1
            
            # 更新期间最大收益
            daily_max_profit = (row['high_D'] / entry_price) - 1
            if daily_max_profit > max_profit_pct:
                max_profit_pct = daily_max_profit
                days_to_max_profit = day_num

            # 更新期间最大回撤
            daily_max_drawdown = (row['low_D'] / entry_price) - 1
            if daily_max_drawdown < max_drawdown_pct:
                max_drawdown_pct = daily_max_drawdown
                days_to_max_drawdown = day_num

            # 检查是否触发止盈或止损
            hit_target = row['high_D'] >= target_price
            hit_stop = row['low_D'] <= stop_price

            if hit_target and hit_stop:
                # 如果一天内同时触及，判断哪个先到
                # 这是一个简化处理，实战中需要更高频数据。我们假设先到止损。
                exit_reason = 'stop_loss'
                final_outcome = 'failure'
                exit_days = day_num
                break
            elif hit_target:
                exit_reason = 'profit_target'
                final_outcome = 'success'
                exit_days = day_num
                break
            elif hit_stop:
                exit_reason = 'stop_loss'
                final_outcome = 'failure'
                exit_days = day_num
                break
        
        return {
            'outcome': final_outcome,
            'exit_reason': exit_reason,
            'exit_days': exit_days,
            'max_profit_pct': max_profit_pct,
            'days_to_max_profit': days_to_max_profit,
            'max_drawdown_pct': max_drawdown_pct,
            'days_to_max_drawdown': days_to_max_drawdown,
        }

    def _aggregate_and_report_v2(self, trade_outcomes: list) -> list:
        """
        【V3.0 新增核心方法】聚合所有详细交易结果，并按信号来源分组统计。
        """
        if not trade_outcomes:
            return []
            
        outcomes_df = pd.DataFrame(trade_outcomes).set_index('entry_date')
        score_map = self.scoring_params.get('score_type_map', {})
        
        # 按信号名称对所有触发的交易结果进行分组
        signal_groups = {}
        for entry_date, outcome_row in outcomes_df.iterrows():
            # 找出当天激活了哪些信号
            active_signals = self.score_details_df.loc[entry_date]
            active_signals = active_signals[active_signals > 0].index
            
            for signal_name in active_signals:
                if signal_name not in signal_groups:
                    signal_groups[signal_name] = []
                signal_groups[signal_name].append(outcome_row.to_dict())

        # 对每个信号组进行统计分析
        analysis_results = []
        for signal_name, outcomes_list in signal_groups.items():
            if signal_name not in score_map:
                continue
            
            group_df = pd.DataFrame(outcomes_list)
            total_triggers = len(group_df)
            success_count = (group_df['outcome'] == 'success').sum()
            
            # 计算胜率
            win_rate = (success_count / total_triggers) * 100 if total_triggers > 0 else 0
            
            # 计算平均指标
            avg_max_profit = group_df['max_profit_pct'].mean() * 100
            avg_max_drawdown = group_df['max_drawdown_pct'].mean() * 100
            avg_exit_days = group_df['exit_days'].mean()
            
            analysis_results.append({
                'signal_name': signal_name,
                'cn_name': score_map[signal_name].get('cn_name', signal_name),
                'type': score_map[signal_name].get('type', 'unknown').capitalize(),
                'triggers': int(total_triggers),
                'successes': int(success_count),
                'win_rate_pct': round(win_rate, 2),
                'avg_max_profit_pct': round(avg_max_profit, 2),
                'avg_max_drawdown_pct': round(avg_max_drawdown, 2),
                'avg_exit_days': round(avg_exit_days, 1),
            })
            
        # 按触发次数排序，方便查看
        analysis_results.sort(key=lambda x: x['triggers'], reverse=True)
        return analysis_results
