# 文件: strategies/trend_following/performance_analyzer.py
# 升级模块：性能分析器 (V3.0 深度复盘版)
from typing import Dict
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
    def __init__(self, df_indicators: pd.DataFrame, score_details_df: pd.DataFrame, 
                 atomic_states: Dict, trigger_events: Dict, 
                 analysis_params: dict, scoring_params: dict):
        """
        【V4.3 构造函数兼容版】
        初始化分析器
        :param df_indicators: 包含最终信号和K线数据的主DataFrame。
        :param score_details_df: 包含每日各信号得分详情的DataFrame。
        :param atomic_states: 包含所有原子状态的字典。
        :param trigger_events: 包含所有触发事件的字典。
        :param analysis_params: 性能分析模块的专属配置。
        :param scoring_params: 四层计分模型的配置，用于获取信号元数据。
        """
        self.df = df_indicators
        self.score_details_df = score_details_df
        self.atomic_states = atomic_states if atomic_states is not None else {}
        self.trigger_events = trigger_events if trigger_events is not None else {}
        self.analysis_params = analysis_params
        self.scoring_params = scoring_params
        if self.df is None or self.df.empty:
            raise ValueError("PerformanceAnalyzer 接收到的 df_indicators 为空。")
        
        # 从配置中获取分析参数
        self.look_forward_days = get_param_value(self.analysis_params.get('look_forward_days'), 20)
        self.profit_target_pct = get_param_value(self.analysis_params.get('profit_target_pct'), 0.15)
        self.stop_loss_pct = get_param_value(self.analysis_params.get('stop_loss_pct'), 0.07)

    def run_analysis(self) -> list:
        """
        【V4.0 重构版】运行性能分析的主函数。
        - 核心重构: 不再只分析买入信号，而是分析所有识别出的“事件”。
        """
        # --- 代码修改开始 ---
        # [修改原因] 全面重构分析流程，以适应对所有信号源的评估。
        print("    -> [性能分析器 V4.0 全景沙盘版] 启动...")
        
        # 步骤1: 识别出所有需要分析的事件
        all_events_to_analyze = self._identify_all_events()
        if not all_events_to_analyze:
            print("      -> 未发现任何可供分析的事件。")
            return []
            
        # 步骤2: 遍历每一个事件，模拟其后续表现
        all_trade_outcomes = []
        for signal_name, event_series in all_events_to_analyze.items():
            event_dates = event_series.index[event_series]
            if event_dates.empty:
                continue
            
            for entry_date in event_dates:
                outcome_details = self._analyze_single_trade_performance(entry_date)
                if outcome_details:
                    all_trade_outcomes.append({
                        'signal_name': signal_name, # 关键：记录下是哪个信号触发的
                        'entry_date': entry_date,
                        **outcome_details
                    })
        
        # 步骤3: 聚合所有结果并生成报告
        return self._aggregate_and_report_v2(all_trade_outcomes)

    def _identify_all_events(self) -> Dict[str, pd.Series]:
        """
        【V4.0 新增】全情报源事件识别器
        - 核心职责: 遍历所有信号源（得分信号、原子状态、触发器），
                    并为每一个源生成一个“首次触发”的事件序列。
        - 核心逻辑: 对于持续性状态，只在其首次变为True的那一天标记为事件。
        """
        all_events = {}
        
        # 1. 处理所有计分信号 (来自 score_details_df)
        for signal_name in self.score_details_df.columns:
            # 计分信号本身就是瞬时事件，直接使用
            is_active = self.score_details_df[signal_name] > 0
            all_events[signal_name] = is_active

        # 2. 处理所有原子状态
        for state_name, state_series in self.atomic_states.items():
            if state_series.dtype == bool:
                # 关键逻辑：只在状态首次进入时触发事件
                is_first_day = state_series & ~state_series.shift(1).fillna(False)
                all_events[state_name] = is_first_day

        # 3. 处理所有触发器
        for trigger_name, trigger_series in self.trigger_events.items():
            if trigger_series.dtype == bool:
                # 触发器本身就是瞬时事件，直接使用
                all_events[trigger_name] = trigger_series
                
        return all_events

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
        【V4.0 升级版】聚合所有详细交易结果，并按信号来源分组统计。
        - 核心升级: 能够识别并正确标记 State 和 Trigger 类型。
        """
        if not trade_outcomes:
            return []
            
        # --- 代码修改开始 ---
        # [修改原因] 不再需要设置 entry_date 为索引，因为 signal_name 现在是分组的关键。
        outcomes_df = pd.DataFrame(trade_outcomes)
        score_map = self.scoring_params.get('score_type_map', {})
        
        # 按信号名称对所有触发的交易结果进行分组
        signal_groups = outcomes_df.groupby('signal_name')

        # 对每个信号组进行统计分析
        analysis_results = []
        for signal_name, group_df in signal_groups:
            # 尝试从 score_map 获取元数据
            signal_meta = score_map.get(signal_name)
            
            # 如果在 score_map 中找不到，则根据其来源（原子状态/触发器）赋予默认类型
            if not signal_meta:
                if signal_name in self.atomic_states:
                    signal_type = 'State'
                elif signal_name in self.trigger_events:
                    signal_type = 'Trigger'
                else:
                    signal_type = 'Unknown'
                signal_meta = {'cn_name': signal_name, 'type': signal_type}
            # --- 代码修改结束 ---

            total_triggers = len(group_df)
            success_count = (group_df['outcome'] == 'success').sum()
            
            win_rate = (success_count / total_triggers) * 100 if total_triggers > 0 else 0
            avg_max_profit = group_df['max_profit_pct'].mean() * 100
            avg_max_drawdown = group_df['max_drawdown_pct'].mean() * 100
            avg_exit_days = group_df['exit_days'].mean()
            
            analysis_results.append({
                'signal_name': signal_name,
                'cn_name': signal_meta.get('cn_name', signal_name),
                'type': signal_meta.get('type', 'unknown').capitalize(),
                'triggers': int(total_triggers),
                'successes': int(success_count),
                'win_rate_pct': round(win_rate, 2),
                'avg_max_profit_pct': round(avg_max_profit, 2),
                'avg_max_drawdown_pct': round(avg_max_drawdown, 2),
                'avg_exit_days': round(avg_exit_days, 1),
            })
            
        analysis_results.sort(key=lambda x: x['win_rate_pct'], reverse=True) # 按胜率排序
        return analysis_results