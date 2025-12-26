# 文件: strategies/multi_timeframe_trend_strategy.py
# 版本: V203.0 总指挥重构版
import re
from datetime import datetime, time
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import json
import os
import pandas as pd
import numpy as np
import traceback
import gc
from services.indicator_services import IndicatorService
from stock_models.index import TradeCalendar
from strategies.trend_following_strategy import TrendFollowStrategy
from strategies.prophet_signal_strategy import ProphetSignalStrategy
from strategies.weekly_context_engine import WeeklyContextEngine
from utils.cache_manager import CacheManager
from utils.config_loader import load_strategy_config
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score


# 初始化日志记录器
logger = logging.getLogger(__name__)

class MultiTimeframeTrendStrategy:
    """
    【V203.0 总指挥重构版】
    多时间框架趋势跟踪策略 - 总指挥层 (Orchestrator)。
    核心职责:
    1.  **统一配置管理**: 加载并持有唯一的全局配置文件。
    2.  **引擎初始化**: 初始化下属的各个专业引擎（周线战略、日线战术等）。
    3.  **作战流程编排**: 按照“数据准备 -> 战略分析 -> 战术决策 -> 盘中执行”的顺序，精确调用各引擎。
    4.  **情报融合**: 将不同时间框架（周、日、分钟）的分析结果进行高效、准确的合并与处理。
    5.  **战报生成**: 汇总所有信号，生成标准化的最终记录。
    """
    def __init__(self, cache_manager_instance: CacheManager):
        """
        【V206.0 · 主权配置协议版】初始化总指挥部。
        - 核心革命: 采纳最高指令，在初始化时即完成配置的净化与隔离。
        - 核心逻辑:
          1. 加载主配置和完整的信号字典。
          2. 将信号字典拆分为两份独立的、纯净的配置：一份给 TrendFollow，一份给 Prophet。
          3. 通过依赖注入，将专属配置传递给各自的策略实例。
        - 收益: 实现了“主权独立，配置隔离”，彻底根除了配置污染问题。
        """
        # 步骤1：加载主配置文件和完整的信号字典
        main_config_path = 'config/trend_follow_strategy.json'
        main_config = load_strategy_config(main_config_path)
        config_dir = os.path.dirname(main_config_path)
        dict_path = os.path.join(config_dir, 'signal_dictionary.json')
        original_score_map = {}
        if os.path.exists(dict_path):
            try:
                with open(dict_path, 'r', encoding='utf-8') as f:
                    original_score_map = json.load(f).get('score_type_map', {})
            except Exception as e:
                logger.error(f"加载或解析信号字典 {dict_path} 失败: {e}")
        # 步骤2：执行“主权配置协议”，为每个策略准备专属配置
        import copy
        # 为 TrendFollow 准备纯净配置
        trend_follow_config = copy.deepcopy(main_config)
        trend_follow_score_map = {k: v for k, v in original_score_map.items() if 'PREDICTIVE_' not in k}
        trend_follow_params = trend_follow_config.get('strategy_params', {}).get('trend_follow', {})
        trend_follow_params['score_type_map'] = trend_follow_score_map
        trend_follow_config['strategy_params']['trend_follow'] = trend_follow_params
        # 为 Prophet 准备纯净配置
        prophet_config = copy.deepcopy(main_config)
        prophet_score_map = {k: v for k, v in original_score_map.items() if 'PREDICTIVE_' in k}
        # 先知策略的配置可能在不同的块中，我们直接在顶层注入
        prophet_config.get('strategy_params', {}).get('trend_follow', {})['score_type_map'] = prophet_score_map
        # 步骤3：使用专属配置初始化所有单元
        self.unified_config = main_config # 保留一个完整的副本以备后用
        self.indicator_service = IndicatorService(cache_manager_instance)
        self.strategic_engine = WeeklyContextEngine(config=self.unified_config)
        # 将专属配置注入到各个策略中
        self.tactical_engine = TrendFollowStrategy(self, trend_follow_config)
        self.prophet_engine = ProphetSignalStrategy(self, prophet_config)
        self.daily_analysis_df = None
        self.required_timeframes = self.indicator_service._discover_required_timeframes_from_config(self.unified_config)
    async def run_for_stock(self, stock_code: str, trade_time: Optional[datetime] = None, latest_only: bool = False, start_date_str: Optional[str] = None) -> Tuple[List, List, List, List, List]:
        """
        【总指挥层核心 - V507.1 · 数据流修复版】
        - 核心重构: 极大简化了数据流。现在直接将包含所有时间框架数据的 all_dfs 传递给战术引擎。
        - 本次修复: 修正了对 _run_tactical_engine 返回的嵌套元组的错误处理，解决了 TypeError 和潜在的 IndexError。
        """
        mode_str = "闪电突袭" if latest_only else "全面战役"
        start_info = f", 计算起始于: {start_date_str}" if start_date_str and not latest_only else ""
        print(f"\n🚀 [总指挥层 - {mode_str}] 开始处理股票: {stock_code}, 交易时间: {trade_time}{start_info}")
        # 1. 数据准备: 加载所有需要的时间框架数据
        all_dfs = await self.indicator_service.prepare_data_for_strategy(
            stock_code, self.unified_config, trade_time, latest_only=latest_only
        )
        if not all_dfs or 'D' not in all_dfs or all_dfs['D'].empty:
            print(f"  - [数据引擎] 未能获取 {stock_code} 的日线数据，跳过处理。")
            return ([], [], [], [], [])
        # 修改变量名，更清晰地接收 _run_tactical_engine 的返回结果
        # tactical_results 是一个四元组: ((signals, details, ...), df1, df2, df3)
        tactical_results = await self._run_tactical_engine(stock_code, all_dfs, start_date_str=start_date_str)
        # 从嵌套元组中正确解包出包含5个列表的内部元组
        # records_from_tactical 现在是 (list_of_signals, list_of_details, ...)
        records_from_tactical = tactical_results[0]
        # 3. 盘中引擎
        intraday_entry_signals, intraday_entry_details = await self._run_intraday_entry_engine(stock_code, all_dfs)
        risk_alert_signals, risk_alert_details = self._run_intraday_alert_engine(stock_code, all_dfs)
        # 修正拼接逻辑，从内部元组中按索引取值
        all_signals = records_from_tactical[0] + intraday_entry_signals + risk_alert_signals
        all_details = records_from_tactical[1] + intraday_entry_details + risk_alert_details
        if all_signals:
            all_signals.sort(key=lambda x: x.trade_time)
        print(f"🏁 [总指挥层] 完成处理 {stock_code}, 共生成 {len(all_signals)} 条主信号记录。")
        # 修正最终返回值的来源，确保返回5个列表
        return (all_signals, all_details, records_from_tactical[2], records_from_tactical[3], records_from_tactical[4])
    async def _run_tactical_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame], start_date_str: Optional[str] = None) -> Tuple[List, List, List, List, List]:
        """
        【V511.0 · 双子星协议版】
        - 核心革命: 作为联邦总指挥，协调两大主权策略的运行与战报合并。
        - 核心逻辑:
          1. 运行“趋势跟踪”引擎，获取其结果，并捕获其计算出的 atomic_states。
          2. 将 atomic_states 作为情报，传递给“先知”引擎，获取其独立结果。
          3. 合并两套战报，形成统一的最终报告。
        """
        try:
            # 步骤1: 运行“趋势跟踪”引擎，获取其结果和计算出的情报
            daily_analysis_df, score_details_df, risk_details_df = self.tactical_engine.apply_strategy(
                all_dfs, start_date_str=start_date_str
            )
            if daily_analysis_df is None or daily_analysis_df.empty:
                return (([], [], [], [], []), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
            trend_follow_records = await self.tactical_engine.prepare_db_records(
                stock_code=stock_code,
                result_df=daily_analysis_df,
                score_details_df=score_details_df,
                risk_details_df=risk_details_df,
                params=self.unified_config,
                result_timeframe='D'
            )
            # 步骤2: 运行“先知”引擎，并将主引擎的情报(atomic_states)传递给它
            prophet_records = await self.prophet_engine.apply_strategy(
                stock_code,
                all_dfs['D'],
                self.tactical_engine.atomic_states
            )
            # 步骤3: 合并两大主权策略的战报
            all_signals = trend_follow_records[0] + prophet_records[0]
            all_details = trend_follow_records[1] + prophet_records[1]
            all_daily_scores = trend_follow_records[2] + prophet_records[2]
            all_score_components = trend_follow_records[3] + prophet_records[3]
            all_daily_states = trend_follow_records[4] + prophet_records[4]
            combined_records = (all_signals, all_details, all_daily_scores, all_score_components, all_daily_states)
            return (combined_records, daily_analysis_df, score_details_df, risk_details_df)
        except Exception as e:
            logger.error(f"在 {stock_code} 的战术引擎执行期间发生错误: {e}", exc_info=True)
            return (([], [], [], [], []), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    async def run_for_latest_signal(self, stock_code: str, trade_time: Optional[datetime] = None) -> Tuple[List, List, List, List, List]:
        """
        【V506.0 适配版 - 闪电突袭模式】
        - 返回值变更: 现在返回一个包含四类对象的元组。
        """
        print(f"\n⚡️ [总指挥层] 接到“闪电突袭”指令，正在以高效模式处理: {stock_code}")
        # run_for_stock 现在返回一个四元组
        all_signals, all_details, all_daily_scores, all_score_components, all_daily_states = await self.run_for_stock(stock_code, trade_time, latest_only=True)
        if not all_signals and not all_daily_scores:
            print(f"  - [闪电突袭] 未发现任何信号或分数，任务完成。")
            return ([], [], [], [], [])
        # 筛选最新的 TradingSignal
        latest_signals = []
        if all_signals:
            latest_date = max(rec.trade_time.date() for rec in all_signals)
            latest_signals = [rec for rec in all_signals if rec.trade_time.date() == latest_date]
        # 筛选最新的 SignalPlaybookDetail
        latest_details = [d for d in all_details if d.signal in latest_signals]
        # 筛选最新的 StrategyDailyScore
        latest_daily_scores = []
        if all_daily_scores:
            latest_date = max(score.trade_date for score in all_daily_scores)
            latest_daily_scores = [score for score in all_daily_scores if score.trade_date == latest_date]
        # 筛选最新的 StrategyScoreComponent
        latest_score_components = [comp for comp in all_score_components if comp.daily_score in latest_daily_scores]
        latest_daily_states = [state for state in all_daily_states if state.daily_score in latest_daily_scores]
        print(f"🏁 [总指挥层-闪电突袭] 高效模式处理完毕, 共生成 {len(latest_signals)} 条最新信号和 {len(latest_daily_scores)} 条最新分数。")
        return (latest_signals, latest_details, latest_daily_scores, latest_score_components, latest_daily_states)
    async def _run_intraday_entry_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> Tuple[List, List]:
        """
        【V203.6 修正版】盘中入场确认引擎
        - 核心修正: 修正了对 get_params_block 和 get_param_value 的调用方式。
        """
        # 导入新模型
        from stock_models.stock_analytics import TradingSignal
        entry_params = get_params_block(self.tactical_engine, 'intraday_entry_params')
        get_val = get_param_value
        if not get_val(entry_params.get('enabled'), False): return ([], [])
        if self.daily_analysis_df is None or self.daily_analysis_df.empty: return ([], [])
        minute_tf = str(get_val(entry_params.get('timeframe'), '5'))
        minute_df = all_dfs.get(minute_tf)
        if minute_df is None or minute_df.empty: return ([], [])
        daily_score_threshold = get_val(entry_params.get('daily_score_threshold'), 100)
        setup_days_df = self.daily_analysis_df[self.daily_analysis_df['entry_score'] >= daily_score_threshold].copy()
        if setup_days_df.empty: return []
        setup_days_df['setup_date'] = setup_days_df.index.date
        trade_dates_series = pd.Series(await TradeCalendar.get_trade_dates_in_range_async(
            start_date=setup_days_df.index.min().date(),
            end_date=(setup_days_df.index.max() + pd.Timedelta(days=5)).date()
        ))
        date_map = pd.Series(trade_dates_series.iloc[1:].values, index=trade_dates_series.iloc[:-1].values)
        setup_days_df['monitoring_date'] = setup_days_df['setup_date'].map(date_map)
        setup_days_df.dropna(subset=['monitoring_date'], inplace=True)
        if setup_days_df.empty: return []
        context_cols = ['monitoring_date', 'entry_score', 'PLATFORM_PRICE_STABLE']
        existing_context_cols = [col for col in context_cols if col in setup_days_df.columns]
        minute_df_with_ts = minute_df.reset_index().rename(columns={'index': 'trade_time'})
        minute_df_with_ts['monitoring_date'] = minute_df_with_ts['trade_time'].dt.date
        merged_minute_df = pd.merge(
            minute_df_with_ts,
            setup_days_df[existing_context_cols],
            on='monitoring_date',
            how='inner'
        )
        if merged_minute_df.empty: return []
        merged_minute_df.set_index('trade_time', inplace=True)
        final_confirmation_signal = pd.Series(True, index=merged_minute_df.index)
        rules = entry_params.get('confirmation_rules', {})
        vwap_rule = rules.get('vwap_reclaim', {})
        if get_val(vwap_rule.get('enabled'), False):
            vwap_col, close_col_m = f'VWAP_{minute_tf}', f'close_{minute_tf}'
            if vwap_col in merged_minute_df.columns and close_col_m in merged_minute_df.columns:
                final_confirmation_signal &= (merged_minute_df[close_col_m] > merged_minute_df[vwap_col])
        vol_rule = rules.get('volume_confirmation', {})
        if get_val(vol_rule.get('enabled'), False):
            vol_ma_col = f'VOL_MA_{get_val(vol_rule.get("ma_period"), 21)}_{minute_tf}'
            volume_col_m = f'volume_{minute_tf}'
            if vol_ma_col in merged_minute_df.columns and volume_col_m in merged_minute_df.columns:
                final_confirmation_signal &= (merged_minute_df[volume_col_m] > merged_minute_df[vol_ma_col])
        min_time_after_open = get_val(rules.get('min_time_after_open'), 15)
        market_open_time = time(9, 30 + min_time_after_open)
        final_confirmation_signal &= (merged_minute_df.index.time >= market_open_time)
        triggered_df = merged_minute_df[final_confirmation_signal]
        if triggered_df.empty: return []
        first_confirmations_df = triggered_df.loc[triggered_df.groupby('monitoring_date').idxmin().iloc[:, 0]]
        final_entry_records = []
        playbook_blueprints = self.tactical_engine.playbook_blueprints
        playbook_cn_map = {p['name']: p.get('cn_name', p['name']) for p in playbook_blueprints}
        final_entry_signals = []
        for timestamp, row in first_confirmations_df.iterrows():
            daily_score = row.get('entry_score', 0)
            bonus_score = get_val(entry_params.get('bonus_score'), 50)
            final_score = daily_score + bonus_score
            # 创建 TradingSignal 对象
            signal_obj = TradingSignal(
                stock_id=stock_code,
                trade_time=timestamp,
                timeframe=minute_tf,
                strategy_name=get_val(entry_params.get('strategy_name'), 'INTRADAY_ENTRY_CONFIRMATION'),
                signal_type=TradingSignal.SignalType.BUY,
                entry_score=final_score,
                risk_score=0.0, # 盘中确认信号，风险分为0
                close_price=row.get(f'close_{minute_tf}'),
            )
            final_entry_signals.append(signal_obj)
        # 返回一个元组，主信号列表在前，空的详情列表在后
        return (final_entry_signals, [])
    def _run_intraday_alert_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> Tuple[List, List]:
        """
        【V203.6 修正版】盘中风险预警引擎
        - 核心修正: 修正了对 get_params_block 和 get_param_value 的调用方式。
        """
        # 导入新模型
        from stock_models.stock_analytics import TradingSignal
        exec_params = get_params_block(self.tactical_engine, 'intraday_execution_params')
        get_val = get_param_value
        if not get_val(exec_params.get('enabled'), False): return ([], [])
        df_daily = all_dfs.get('D')
        if df_daily is None or df_daily.empty: return ([], [])
        minute_tf = str(get_val(exec_params.get('timeframe'), '30'))
        minute_df = all_dfs.get(minute_tf)
        if minute_df is None or minute_df.empty: return ([], [])
        rules_container = exec_params.get('rules', {})
        upthrust_params = rules_container.get('upthrust_rejection', {})
        if not get_val(upthrust_params.get('enabled'), False): return []
        upthrust_calc_params = get_params_block(self.tactical_engine, 'exit_strategy_params').get('upthrust_distribution_params', {})
        lookback_days = get_val(upthrust_calc_params.get('upthrust_lookback_days'), 5)
        is_upthrust_day = df_daily['high_D'] > df_daily['high_D'].shift(1).rolling(window=lookback_days, min_periods=1).max()
        setup_days_df = df_daily[is_upthrust_day].copy()
        if setup_days_df.empty: return []
        setup_days_df['monitoring_date'] = (setup_days_df.index + pd.Timedelta(days=1)).date
        minute_df_with_ts = minute_df.reset_index().rename(columns={'index': 'trade_time'})
        minute_df_with_ts['monitoring_date'] = minute_df_with_ts['trade_time'].dt.date
        merged_minute_df = pd.merge(minute_df_with_ts, setup_days_df[['monitoring_date']], on='monitoring_date', how='inner')
        if merged_minute_df.empty: return []
        merged_minute_df.set_index('trade_time', inplace=True)
        close_col, vwap_col = f'close_{minute_tf}', f'VWAP_{minute_tf}'
        if vwap_col not in merged_minute_df.columns or close_col not in merged_minute_df.columns: return []
        is_breaking_down = merged_minute_df[close_col] < merged_minute_df[vwap_col]
        first_breakdown_signal = is_breaking_down & ~is_breaking_down.shift(1).fillna(False)
        alert_days = merged_minute_df[first_breakdown_signal]['monitoring_date'].unique()
        if len(alert_days) == 0: return []
        def process_alert_day(day_df: pd.DataFrame) -> Optional[Dict]:
            is_breaking = day_df[close_col] < day_df[vwap_col]
            first_break_mask = is_breaking & ~is_breaking.shift(1).fillna(False)
            if not first_break_mask.any(): return None
            first_break_timestamp = first_break_mask.idxmax()
            first_alert_row = day_df.loc[first_break_timestamp]
            df_after_alert = day_df[day_df.index > first_break_timestamp]
            is_reclaimed = (df_after_alert[close_col] > df_after_alert[vwap_col]).any()
            signal_type = '风险预警'
            if is_reclaimed:
                # 威胁解除，不生成信号
                return None
            else:
                final_reason = f"盘中于{first_break_timestamp.strftime('%H:%M')}跌破VWAP且至收盘未收复"
                final_code = get_val(upthrust_params.get('alert_code'), 103)
                # 创建 TradingSignal 对象
                return TradingSignal(
                    stock_id=stock_code,
                    trade_time=first_break_timestamp,
                    timeframe=minute_tf,
                    strategy_name="INTRADAY_RISK_ALERT",
                    signal_type=TradingSignal.SignalType.WARN, # 信号类型为预警
                    entry_score=0.0,
                    risk_score=float(final_code), # 风险分记录警报代码
                    close_price=first_alert_row[close_col],
                    # 可以考虑将 final_reason 存入某个JSON字段，如果模型支持的话
                )
        final_alerts = merged_minute_df[merged_minute_df['monitoring_date'].isin(alert_days)]\
            .groupby('monitoring_date', group_keys=False)\
            .apply(process_alert_day)\
            .dropna().tolist()
        # 返回元组
        return (final_alerts, [])
    # ▼▼▼ 报告生成函数重大升级，以支持分级止盈 ▼▼▼
    def _generate_analysis_report(self, record: Dict[str, Any]) -> str:
        stock_code = record.get("stock_code", "N/A")
        trade_time = record.get("trade_time")
        time_str = trade_time.strftime('%Y-%m-%d %H:%M:%S') if isinstance(trade_time, datetime) else str(trade_time)
        timeframe = record.get("timeframe", "N/A")
        report_parts = [f"*** 信号分析报告 ({stock_code}) ***"]
        report_parts.append(f"信号时间: {time_str} (周期: {timeframe})")
        if record.get('exit_signal_code', 0) > 0:
            severity = record.get('exit_severity_level', 2) # 默认为二级
            reason = record.get('exit_signal_reason', '未定义的原因')
            if severity == 1: # 一级预警
                report_parts.append("信号类型: 【一级预警·黄色】趋势观察")
                report_parts.append(f"核心发现: **上涨动能出现减弱迹象，但趋势尚未破坏。**")
                report_parts.append(f"触发原因: {reason}")
                report_parts.append("建议操作: 密切关注后续K线，可考虑部分减仓锁定利润，或上移追踪止损位。")
            elif severity == 3: # 三级警报
                report_parts.append("信号类型: 【三级警报·红色】紧急离场")
                report_parts.append(f"核心发现: **上涨结构已被破坏，风险急剧升高！**")
                report_parts.append(f"触发原因: {reason}")
                report_parts.append("建议操作: 立即离场以控制风险，观望为主。")
            else: # 二级警报 (默认)
                report_parts.append("信号类型: 【二级警报·橙色】标准止盈")
                report_parts.append(f"核心发现: **短期趋势确认转弱，已触发标准卖出条件。**")
                report_parts.append(f"触发原因: {reason}")
                report_parts.append("建议操作: 执行止盈计划，建议减仓或清仓。")
        elif record.get('entry_signal', False):
            score = record.get('entry_score', 0.0)
            playbooks = record.get('triggered_playbooks', [])
            report_parts.append(f"信号类型: 综合买入 (总分: {score:.2f})")
            report_parts.append("核心发现: **多个看涨剧本共振，形成高置信度买入信号！**")
            if playbooks:
                report_parts.append("触发剧本:")
                for playbook in sorted(playbooks):
                    report_parts.append(f"  - {playbook}")
        return "\n".join(report_parts)
    # NEW: 新增的性能分析专属方法
    async def analyze_signal_performance_for_period(self, stock_code: str, start_date: str, end_date: str):
        """
        【V1.1 返回值适配版】信号性能分析总指挥方法
        - 方法现在会返回性能分析器计算出的原始结果列表，以供上层调用者（如Celery任务）进行格式化展示。
        - 职责: 作为一个独立的、用于深度回测的入口，编排策略运行和性能分析的流程。
        - 流程:
          1. 运行全历史策略，生成回测区间内的所有指标和信号。
          2. 检查性能分析模块是否在配置中启用。
          3. 如果启用，则实例化性能分析器，并将策略运行结果注入。
          4. 启动分析器，获取并返回分析结果。
        """
        print("=" * 80)
        print(f"--- [信号性能分析任务启动 V1.2] ---")
        print(f"    -> 股票代码: {stock_code}")
        print(f"    -> 分析时段: {start_date} to {end_date}")
        print("=" * 80)
        analysis_results = []
        try:
            # 步骤 1: 运行核心策略，生成回测数据
            print("    -> [阶段 1/3] 正在执行全历史策略计算，请稍候...")
            # 使用四元组接收所有返回结果
            _all_signals, _all_details, _all_daily_scores, _all_score_components = await self.run_for_stock(
                stock_code, trade_time=end_date, latest_only=False
            )
            print("    -> [阶段 1/3] 策略计算完成。")
            # 步骤 2: 检查并准备启动分析器
            print("    -> [阶段 2/3] 正在准备启动性能分析器...")
            analyzer_params = get_params_block(self.tactical_engine, 'performance_analysis_params')
            if not get_param_value(analyzer_params.get('enabled'), False):
                print("    -> [信息] 性能分析模块在配置文件中被禁用，任务终止。")
                return []
            # 从战术引擎获取最后一次运行的详细结果
            df_indicators = self.daily_analysis_df
            score_details_df = getattr(self.tactical_engine, '_last_score_details_df', pd.DataFrame())
            if df_indicators is None or df_indicators.empty or score_details_df.empty:
                print("    -> [错误] 策略运行后未能获取有效的分析数据，无法进行性能分析。")
                return []
            # 步骤 3: 运行分析器
            print("    -> [阶段 3/3] 注入数据并运行分析器...")
            try:
                # 动态导入，保持主模块干净
                from .trend_following.performance_analyzer import PerformanceAnalyzer
                scoring_params = get_params_block(self.tactical_engine, 'four_layer_scoring_params')
                analyzer = PerformanceAnalyzer(
                    df_indicators=df_indicators,
                    score_details_df=score_details_df,
                    atomic_states=self.tactical_engine.atomic_states,
                    trigger_events=self.tactical_engine.trigger_events,
                    playbook_states=self.tactical_engine.playbook_states,
                    analysis_params=analyzer_params,
                    scoring_params=scoring_params
                )
                # 捕获分析器返回的原始数据
                analysis_results = analyzer.run_analysis()
            except ImportError:
                print("    -> [严重错误] 无法导入 PerformanceAnalyzer 模块。请确保文件存在于 'strategies/trend_following/' 目录下。")
            except Exception as e:
                print(f"    -> [严重错误] 性能分析器在执行过程中发生异常: {e}")
                traceback.print_exc()
            print(f"--- [信号性能分析任务完成] ---")
        except Exception as e:
            print(f"[严重错误] 在执行信号性能分析时发生顶层异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 手动清理大型DataFrame，释放内存
            self.daily_analysis_df = None
            if hasattr(self.tactical_engine, '_last_score_details_df'):
                del self.tactical_engine._last_score_details_df
            if hasattr(self.tactical_engine, '_last_risk_details_df'):
                del self.tactical_engine._last_risk_details_df
            gc.collect()
            print("    -> [内存管理] 已清理本次分析任务产生的临时数据。")
            # 在finally块中返回结果，确保无论如何都有返回值
            return analysis_results
    async def debug_run_for_period(self, stock_code: str, start_date: str, end_date: str):
        """
        【V324.0 · 真理之镜协议版】
        - 核心修复: 修正了探针与最终报告数据源不一致的致命BUG。
        - 核心逻辑:
          1. 在核心策略计算完成后，立即使用返回的最新数据部署法医探针。
          2. 确保探针的解剖报告和最终的信号透视报告，都基于同一次运行的、完全相同的数据。
        - 收益: 实现了调试信息的绝对同步，确保所见即所得，根除了因数据不同步导致的认知混乱。
        """
        print("=" * 80)
        print(f"--- [历史回溯调试启动 V324.0 · 真理之镜协议版] ---") # 更新版本号
        print(f"    -> 股票代码: {stock_code}")
        print(f"    -> 回测时段: {start_date} to {end_date}")
        print("=" * 80)
        try:
            # 步骤 1: 独立执行数据准备和战术引擎，并捕获所有返回结果
            # print("    -> [阶段 1/3] 正在执行核心策略计算，以捕获调试所需数据...")
            all_dfs = await self.indicator_service.prepare_data_for_strategy(stock_code, self.unified_config, end_date, latest_only=False)
            engine_results = await self._run_tactical_engine(
                stock_code, all_dfs, start_date_str=start_date
            )
            if not isinstance(engine_results, tuple) or len(engine_results) < 4:
                print("[严重错误] 战术引擎返回结果格式不正确，无法继续调试。")
                return
            _records_tuple, daily_analysis_df, score_details_df, risk_details_df = engine_results
            if daily_analysis_df is None or daily_analysis_df.empty:
                print("[严重错误] 战术引擎未能生成有效的分析数据(daily_analysis_df)，调试终止。")
                return
            # print("    -> [阶段 1/3] 核心策略计算完成。")
            # 步骤 2: 立即部署探针，确保其在最新的数据上运行
            # print("\n    -> [阶段 2/3] 正在部署法医探针，以解剖本次运行的中间过程...")
            debug_params = get_params_block(self.tactical_engine, 'debug_params')
            # --- 新增的调试打印 ---
            print(f"    -> [Debug Probe Check] 从 get_params_block 获取到的 debug_params: {debug_params}")
            raw_enabled_value = debug_params.get('enabled')
            print(f"    -> [Debug Probe Check] debug_params.get('enabled') 的原始值: {raw_enabled_value}")
            resolved_enabled_value = get_param_value(raw_enabled_value, False)
            print(f"    -> [Debug Probe Check] 经过 get_param_value 解析后的 enabled 值: {resolved_enabled_value}")
            # --- 调试打印结束 ---
            if get_param_value(debug_params.get('enabled'), False):
                # 确保探针使用的是本次运行的最新 atomic_states
                self.tactical_engine.intelligence_layer.deploy_forensic_probes()
            else:
                print("    -> [信息] 法医探针在配置中被禁用，跳过解剖。")
            # 步骤 3: 使用本次运行的、唯一的 daily_analysis_df 生成最终报告
            # print(f"\n    -> [阶段 3/3] 正在筛选并展示目标时段 ({start_date} to {end_date}) 的所有信号和每日分数...")
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            if isinstance(daily_analysis_df.index, pd.DatetimeIndex) and daily_analysis_df.index.tz is not None:
                target_timezone = daily_analysis_df.index.tz
                start_dt = start_dt.tz_localize(target_timezone)
                end_dt = end_dt.tz_localize(target_timezone)
            debug_period_df = daily_analysis_df[(daily_analysis_df.index >= start_dt) & (daily_analysis_df.index <= end_dt)]
            if debug_period_df.empty:
                print(f"[信息] 在指定时段 {start_date} to {end_date} 内没有找到任何分析数据。")
                print(f"    -> 提示: 请检查完整数据(daily_analysis_df)的索引范围是否覆盖此期间。完整数据范围: {daily_analysis_df.index.min()} to {daily_analysis_df.index.max()}")
                return
            print("\n" + "="*30 + " [全流程信号透视报告] " + "="*30)
            for trade_date, row in debug_period_df.iterrows():
                time_str = trade_date.strftime('%Y-%m-%d')
                final_score_val = row.get('final_score', 'N/A')
                signal_type = row.get('signal_type', '无信号')
                final_score_str = f"{final_score_val:<7.0f}" if isinstance(final_score_val, (int, float)) else "N/A"
                print(f"\n{time_str} [最终得分: {final_score_str}] [最终信号: {signal_type}]")
                score_details_json = row.get('signal_details_cn', {})
                if score_details_json and isinstance(score_details_json, dict):
                    offense_details = score_details_json.get('offense', [])
                    if offense_details:
                        print("  --- 激活进攻项 ---")
                        for item in offense_details:
                            if isinstance(item, dict):
                                print(f"    - {item.get('name', 'N/A'):<20} ({item.get('score', 0):>5.0f})")
                    risk_details = score_details_json.get('risk', [])
                    if risk_details:
                        print("  --- 激活风险项 ---")
                        for item in risk_details:
                            if isinstance(item, dict):
                                print(f"    - {item.get('name', 'N/A'):<20} ({item.get('score', 0):>5.0f})")
            # print(f"\n--- [历史回溯调试完成] ---")
        except Exception as e:
            print(f"[严重错误] 在执行历史回溯调试时发生顶层异常: {e}")
            traceback.print_exc()
        finally:
            gc.collect()
            # print("    -> [内存管理] 已清理本次分析任务产生的临时数据。")
    def _deploy_bottom_reversal_probe(self, probe_date: str, daily_analysis_df: pd.DataFrame, atomic_states: dict):
        """
        【V1.3 · 作用域及逻辑修复版】底部反转信号深度诊断探针
        - 核心修复:
          - [BUG修复] 移除了方法内部的 `import normalize_score` 语句，解决了 UnboundLocalError。
          - [逻辑修复] 修正了探针1中的打印逻辑，使其与新的、基于MA55的上下文计算逻辑保持一致，并修复了多个变量未定义的BUG。
        """
        print("\n" + "="*35 + f" [底部反转信号探针 V1.3] 正在解剖 {probe_date} " + "="*35) # 更新版本号
        try:
            df = daily_analysis_df
            probe_ts = pd.to_datetime(probe_date)
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz: probe_ts = probe_ts.tz_localize(df.index.tz)
            if probe_ts not in df.index:
                print(f"  [错误] 探针日期 {probe_date} 不在数据范围内。")
                return
            # --- 探针 1: 修复了所有变量未定义和逻辑不匹配的问题 ---
            print("\n--- [探针 1/3] 解剖：底部情景分 (Context Score) ---")
            ma55 = df.get('EMA_55_D', df['close_D'])
            rolling_high_55d = df['high_D'].rolling(window=55, min_periods=21).max()
            wave_channel_height_top = (rolling_high_55d - ma55).replace(0, 1e-9)
            top_context_score = ((df['close_D'] - ma55) / wave_channel_height_top).clip(0, 1).fillna(0.5)
            price_pos_score = 1 - top_context_score
            rsi_w_oversold_score = normalize_score(df.get('RSI_13_W', pd.Series(50, index=df.index)), df.index, window=52, ascending=False, default_value=0.5)
            cycle_phase = self.tactical_engine.atomic_states.get('DOMINANT_CYCLE_PHASE', pd.Series(0.0, index=df.index)).fillna(0.0)
            cycle_trough_score = (1 - cycle_phase) / 2.0
            bottom_context_score_values = np.maximum.reduce([price_pos_score.values, rsi_w_oversold_score.values, cycle_trough_score.values])
            bottom_context_score = pd.Series(bottom_context_score_values, index=df.index)
            # 提取探针当日的精确值用于打印
            close_price_val = df.get('close_D', pd.Series()).get(probe_ts, 0.0)
            ma55_val = ma55.get(probe_ts, 0.0)
            rolling_high_55d_val = rolling_high_55d.get(probe_ts, 0.0)
            top_context_score_val = top_context_score.get(probe_ts, 0.0)
            bottom_context_score_val = bottom_context_score.get(probe_ts, 0.0)
            print(f"  - 当日收盘价: {close_price_val:.2f}")
            print(f"  - 波段下轨 (MA55): {ma55_val:.2f}")
            print(f"  - 波段上轨 (55日高点): {rolling_high_55d_val:.2f}")
            print(f"  - 价格在波段伸展度: {top_context_score_val:.2%}")
            print(f"  - ✅ 最终底部情景分 (Context): {bottom_context_score_val:.4f}")
            # --- 探针 2: 核心逻辑重构，适配“奖励模式” ---
            print("\n--- [探针 2/3] 解剖：整体看涨反转触发分 (Trigger Score) ---")
            print("  -> 采用“奖励模式”公式进行反推: Trigger = Final Score / (1 + Context * Bonus Factor)")
            p_chip_conf = get_params_block(self.tactical_engine, 'chip_ultimate_params', {})
            bonus_factor = get_param_value(p_chip_conf.get('bottom_context_bonus_factor'), 0.5)
            print(f"  -> 使用的奖励因子 (Bonus Factor): {bonus_factor}")
            engine_prefixes = ['CHIP', 'DYN', 'STRUCTURE', 'BEHAVIOR', 'FF', 'FOUNDATION']
            all_trigger_scores = {}
            for prefix in engine_prefixes:
                signal_name = f"SCORE_{prefix}_BOTTOM_REVERSAL_S_PLUS"
                if signal_name in atomic_states:
                    final_score = atomic_states[signal_name].get(probe_ts, 0.0)
                    denominator = (1 + bottom_context_score_val * bonus_factor)
                    trigger_score = final_score / denominator if denominator > 0 else 0
                    all_trigger_scores[prefix] = trigger_score
                    print(f"  - {prefix:<12s} | 最终分: {final_score:.4f} | 反推触发分: {trigger_score:.4f}")
            avg_trigger_score = np.nanmean(list(all_trigger_scores.values()))
            print(f"  - ✅ 平均触发分 (估算): {avg_trigger_score:.4f}")
            # --- 探针 3: 移除了内部的 import 语句 ---
            print("\n--- [探针 3/3] 深入解剖：以筹码集中度的动态分 (5日周期) 为例 ---")
            slope_raw = df.get(f'SLOPE_5_concentration_90pct_D', pd.Series(0, index=df.index)).get(probe_ts, 0.0)
            accel_raw = df.get(f'ACCEL_5_concentration_90pct_D', pd.Series(0, index=df.index)).get(probe_ts, 0.0)
            # from .trend_following.utils import normalize_score # [代码删除] 移除此行
            slope_norm = normalize_score(df[f'SLOPE_5_concentration_90pct_D'], df.index, 120, ascending=False).get(probe_ts, 0.5)
            accel_norm = normalize_score(df[f'ACCEL_5_concentration_90pct_D'], df.index, 120, ascending=False).get(probe_ts, 0.5)
            dynamic_health_conc = slope_norm * 0.6 + accel_norm * 0.4
            print(f"  - 5日集中度斜率 (原始值): {slope_raw:.4f}")
            print(f"  - 5日集中度斜率 (归一化): {slope_norm:.4f}")
            print(f"  - 5日集中度加速度 (原始值): {accel_raw:.4f}")
            print(f"  - 5日集中度加速度 (归一化): {accel_norm:.4f}")
            print(f"  - ✅ 集中度动态健康分 (估算): {dynamic_health_conc:.4f}")
            print("\n" + "="*35 + " [底部反转信号探针] 解剖完毕 " + "="*35 + "\n")
        except Exception as e:
            print(f"  [探针错误] 在执行“底部反转信号探针”时发生异常: {e}")
            import traceback
            traceback.print_exc()












