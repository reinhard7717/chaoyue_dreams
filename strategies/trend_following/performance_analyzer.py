# 文件: strategies/trend_following/performance_analyzer.py
# 升级模块：性能分析器 (V4.7 法典统一版)
from typing import Dict
import pandas as pd
import numpy as np
from strategies.trend_following.utils import get_param_value, get_params_block # 导入 get_params_block
from services.performance_analysis_service import PerformanceAnalysisService

class PerformanceAnalyzer:
    """
    【V4.7 · 法典统一版】策略性能分析器
    - 核心修复: 修复了本模块错误地从一个过时路径加载 score_map 的致命BUG。
    - 核心逻辑: 强制使用与所有模块一致的 get_params_block 工具来获取权威的 score_type_map，
                  从而解决了因“影子法典”导致的情报黑洞问题。
    """
    def __init__(self, df_indicators: pd.DataFrame, score_details_df: pd.DataFrame, 
                 atomic_states: Dict, trigger_events: Dict, playbook_states: Dict,
                 analysis_params: dict, scoring_params: dict):
        """
        【V4.4 全信号源构造版】
        初始化分析器
        :param df_indicators: 包含最终信号和K线数据的主DataFrame。
        :param score_details_df: 包含每日各信号得分详情的DataFrame。
        :param atomic_states: 包含所有原子状态的字典。
        :param trigger_events: 包含所有触发事件的字典。
        :param playbook_states: 包含所有战法剧本激活状态的字典。
        :param analysis_params: 性能分析模块的专属配置。
        :param scoring_params: 四层计分模型的配置，用于获取信号元数据。
        """
        self.df = df_indicators
        self.score_details_df = score_details_df
        self.atomic_states = atomic_states if atomic_states is not None else {}
        self.trigger_events = trigger_events if trigger_events is not None else {}
        self.playbook_states = playbook_states if playbook_states is not None else {}
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
        【V4.7 · 法典统一版】运行性能分析的主函数。
        - 核心修复: 使用 get_params_block 获取 score_map，确保与系统其他部分一致。
        """
        # print("    -> [性能分析器 V4.7 法典统一版] 启动...")
        # 步骤1: 识别出所有需要分析的事件
        all_events_to_analyze = self._identify_all_events()
        if not all_events_to_analyze:
            print("      -> 未发现任何可供分析的事件。")
            return []
        # 步骤2: 遍历每一个事件，模拟其后续表现
        all_trade_outcomes = []
        # 使用健壮的 get_params_block 获取权威的 score_map
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        for signal_name, event_series in all_events_to_analyze.items():
            event_dates = event_series.index[event_series]
            if event_dates.empty:
                continue
            # [核心逻辑] 查找信号元数据以确定其角色
            signal_meta = score_map.get(signal_name, {})
            signal_type = signal_meta.get('type', 'positional').lower() # 默认为进攻型
            is_offensive_signal = (signal_type != 'risk')
            for entry_date in event_dates:
                # [核心逻辑] 将信号角色传递给模拟函数
                outcome_details = self._analyze_single_trade_performance(entry_date, is_offensive=is_offensive_signal)
                if outcome_details:
                    all_trade_outcomes.append({
                        'signal_name': signal_name,
                        'entry_date': entry_date,
                        **outcome_details
                    })
        # 步骤3: 聚合所有结果并生成报告
        return self._aggregate_and_report_v2(all_trade_outcomes)
    def _identify_all_events(self) -> Dict[str, pd.Series]:
        """
        【V4.6 终极净化版】全情报源事件识别器
        - 核心修复: 修复了过滤器漏洞，现在对所有来源的信号（包括计分详情）
                    都应用统一的、严格的过滤规则。
        """
        all_events = {}
        # 步骤1: 收集所有潜在的信号源
        # 1.1 来自计分详情
        for signal_name in self.score_details_df.columns:
            is_active = self.score_details_df[signal_name] > 0
            all_events[signal_name] = is_active
        # 1.2 来自原子状态
        for state_name, state_series in self.atomic_states.items():
            if state_series.dtype == bool:
                is_first_day = state_series & ~state_series.shift(1).fillna(False)
                all_events[state_name] = is_first_day
        # 1.3 来自触发器
        for trigger_name, trigger_series in self.trigger_events.items():
            if trigger_series.dtype == bool:
                all_events[trigger_name] = trigger_series
        # 1.4 来自战法剧本
        for playbook_name, playbook_series in self.playbook_states.items():
            if playbook_series.dtype == bool:
                all_events[playbook_name] = playbook_series
        # --- 开始：应用统一的终极过滤器 ---
        # 使用健壮的 get_params_block 获取权威的 score_map
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        filtered_events = {}
        print("      -> [战报净化系统 V4.6 终极版] 启动过滤...")
        # 步骤2: 对所有收集到的信号进行统一过滤
        for signal_name, signal_series in all_events.items():
            signal_meta = score_map.get(signal_name)
            # 核心过滤逻辑：
            # 1. 必须在 score_type_map 中有定义
            # 2. 定义中必须有 'type' 键
            # 3. 'type' 的值不能是 'context'
            if signal_meta and 'type' in signal_meta and signal_meta['type'] != 'context':
                filtered_events[signal_name] = signal_series
            else:
                # 调试信息：打印被过滤掉的信号及其原因
                if not signal_meta:
                    reason = "原因: 在 score_type_map 中未定义"
                elif signal_meta.get('type') == 'context':
                    reason = "原因: 类型为 'context'"
                else:
                    reason = "原因: 元数据格式不完整"
                # print(f"          - 已过滤信号: {signal_name} ({reason})")
        original_count = len(all_events)
        filtered_count = len(filtered_events)
        print(f"      -> [战报净化] 已执行过滤：从 {original_count} 个原始信号中筛选出 {filtered_count} 个战斗/风险信号进行分析。")
        return filtered_events
    def _analyze_single_trade_performance(self, entry_date, is_offensive: bool) -> dict:
        """
        【V4.1 角色识别版】深度分析单次交易的性能表现。
        - 接收 is_offensive 参数，以决定调用哪种模拟逻辑。
        """
        # 调用权威的静态模拟函数，并传入正确的信号角色。
        return PerformanceAnalysisService._simulate_trade_outcome(
            entry_date=entry_date,
            price_df=self.df,
            look_forward_days=self.look_forward_days,
            profit_target_pct=self.profit_target_pct,
            stop_loss_pct=self.stop_loss_pct,
            is_offensive=is_offensive # 将接收到的角色参数传递下去
        )
    def _aggregate_and_report_v2(self, trade_outcomes: list) -> list:
        """
        【V4.7 · 法典统一版】
        - 核心修复: 使用 get_params_block 获取 score_map，确保与系统其他部分一致。
        """
        if not trade_outcomes:
            return []
        outcomes_df = pd.DataFrame(trade_outcomes)
        # 使用健壮的 get_params_block 获取权威的 score_map
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        signal_groups = outcomes_df.groupby('signal_name')
        analysis_results = []
        for signal_name, group_df in signal_groups:
            signal_meta = score_map.get(signal_name, {})
            signal_cn_name = signal_meta.get('cn_name', signal_name)
            # --- 开始：明确获取信号类型 ---
            signal_type = signal_meta.get('type', 'unknown')
            total_triggers = len(group_df)
            if total_triggers == 0:
                continue
            success_count = (group_df['outcome'] == 'success').sum()
            # --- 引入分类评估指标 ---
            metric_name = "win_rate" # 默认指标名称
            if signal_type == 'risk':
                # 对于风险信号，计算“风险规避率”
                effectiveness_pct = ((total_triggers - success_count) / total_triggers) if total_triggers > 0 else 0
                metric_name = "avoidance_rate" # 指标名称改为风险规避率
            else:
                # 对于进攻型信号，计算“胜率”
                effectiveness_pct = (success_count / total_triggers) if total_triggers > 0 else 0
            # --- 结束 ---
            avg_max_profit = group_df['max_profit_pct'].mean()
            avg_max_drawdown = group_df['max_drawdown_pct'].mean()
            avg_exit_days = group_df['exit_days'].mean()
            result_entry = {
                'signal_name': signal_name,
                'signal_cn_name': signal_cn_name,
                'signal_type': signal_type,
                'total_triggers': int(total_triggers),
                'successes': int(success_count),
                'effectiveness_pct': effectiveness_pct, # 标准化效能指标
                'metric_name': metric_name,             # 效能指标的名称
                'avg_max_profit_pct': avg_max_profit,
                'avg_max_drawdown_pct': avg_max_drawdown,
                'avg_exit_days': avg_exit_days,
            }
            analysis_results.append(result_entry)
        # 按效能指标降序排序
        analysis_results.sort(key=lambda x: x['effectiveness_pct'], reverse=True)
        return analysis_results











