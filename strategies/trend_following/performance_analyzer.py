# 文件: strategies/trend_following/performance_analyzer.py
# 升级模块：性能分析器 (V2.1 MapReduce适配版)
import pandas as pd
from .utils import get_param_value, get_params_block

class PerformanceAnalyzer:
    """
    【V2.1 MapReduce适配版】策略性能分析器
    - 核心修改: run_analysis 方法不再打印报告，而是返回原始的统计结果列表，
                以支持分布式计算后的聚合操作。
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
        self.look_forward_days = get_param_value(self.analysis_params.get('look_forward_days'), 20)
        self.profit_target_pct = get_param_value(self.analysis_params.get('profit_target_pct'), 0.15)
        self.stop_loss_pct = get_param_value(self.analysis_params.get('stop_loss_pct'), 0.07)

    def run_analysis(self) -> list: # MODIFIED: 修改返回类型提示
        """
        运行性能分析的主函数，并返回结果列表
        """
        # print("\n--- [性能分析模块 V2.1] 启动，开始进行战后复盘... ---") # 在大规模任务中注释掉，减少日志噪音
        # print(f"    -> 分析参数: 向前看 {self.look_forward_days} 天, 盈利目标 {self.profit_target_pct*100:.1f}%, 止损线 {self.stop_loss_pct*100:.1f}%")
        buy_signals = self.df[self.df['signal_type'] == '买入信号']
        if buy_signals.empty:
            return [] # MODIFIED: 返回空列表
        # print(f"    -> 发现 {len(buy_signals)} 个买入信号，正在逐一分析其后续表现...")
        trade_outcomes = []
        for entry_date, row in buy_signals.iterrows():
            outcome = self._simulate_trade_outcome(entry_date)
            trade_outcomes.append({
                'entry_date': entry_date,
                'outcome': outcome
            })
        # MODIFIED: 调用聚合方法并直接返回其结果
        return self._aggregate_and_report(trade_outcomes)

    def _simulate_trade_outcome(self, entry_date) -> str:
        """
        模拟单次交易的最终结果 (成功/失败/超时)
        """
        entry_price = self.df.loc[entry_date, 'close_D']
        target_price = entry_price * (1 + self.profit_target_pct)
        stop_price = entry_price * (1 - self.stop_loss_pct)
        try:
            entry_idx = self.df.index.get_loc(entry_date)
            look_forward_df = self.df.iloc[entry_idx + 1 : entry_idx + 1 + self.look_forward_days]
        except (KeyError, IndexError):
            return 'no_data'
        if look_forward_df.empty:
            return 'no_data'
        for date, row in look_forward_df.iterrows():
            if row['high_D'] >= target_price:
                return 'success'
            if row['low_D'] <= stop_price:
                return 'failure'
        return 'timeout'

    def _aggregate_and_report(self, trade_outcomes: list) -> list: # MODIFIED: 修改返回类型提示
        """
        聚合所有交易结果，并按信号来源进行分组统计，返回结构化数据。
        """
        if not trade_outcomes:
            return []
        outcomes_df = pd.DataFrame(trade_outcomes).set_index('entry_date')
        score_map = self.scoring_params.get('score_type_map', {})
        analysis_results = []
        for signal_name in self.score_details_df.columns:
            if signal_name not in score_map:
                continue
            triggered_days = self.score_details_df.index[self.score_details_df[signal_name] > 0]
            relevant_outcomes = outcomes_df.loc[outcomes_df.index.intersection(triggered_days)]
            if relevant_outcomes.empty:
                continue
            total_triggers = len(relevant_outcomes)
            success_count = (relevant_outcomes['outcome'] == 'success').sum()
            # MODIFIED: 不再计算成功率，只返回原始计数，由最终的聚合任务计算
            analysis_results.append({
                'signal_name': signal_name, # 使用原始名称作为key
                'cn_name': score_map[signal_name].get('cn_name', signal_name),
                'type': score_map[signal_name].get('type', 'unknown').capitalize(),
                'triggers': total_triggers,
                'successes': success_count,
            })
        return analysis_results # MODIFIED: 返回原始数据列表
