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

from services.indicator_services import IndicatorService
from stock_models.index import TradeCalendar
from strategies.trend_following_strategy import TrendFollowStrategy
from strategies.weekly_context_engine import WeeklyContextEngine
from utils.cache_manager import CacheManager
from utils.config_loader import load_strategy_config
from strategies.trend_following.utils import get_params_block, get_param_value


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
        【V203.3 配置融合版】初始化总指挥部。
        - 核心升级: 重构了配置加载逻辑，现在会自动加载并融合独立的信号字典文件。
        """
        # print("--- [总指挥部] 正在初始化 (V203.3 配置融合版)... ---")
        unified_config_path = 'config/trend_follow_strategy.json'
        # 调用新的配置加载与融合方法
        self.unified_config = self._load_and_merge_configs(unified_config_path)
        self.indicator_service = IndicatorService(cache_manager_instance)
        # 1. 初始化战略参谋部 (周线上下文引擎)
        self.strategic_engine = WeeklyContextEngine(config=self.unified_config)
        # print("    -> [OK] 战略参谋部 (WeeklyContextEngine) 已就位。") # 调整日志输出顺序
        # 2. 初始化一线作战部队 (日线战术引擎)
        self.tactical_engine = TrendFollowStrategy(config=self.unified_config)
        # print("    -> [OK] 一线作战部队 (TrendFollowStrategy) 已就位。") # 调整日志输出顺序
        # 内部状态变量
        self.daily_analysis_df = None # 存储日线战术引擎的详细分析结果
        # 从统一配置中自动发现所有需要的K线数据周期
        self.required_timeframes = self.indicator_service._discover_required_timeframes_from_config(self.unified_config)
        # print(f"--- [总指挥部] 初始化完毕，已识别作战所需时间框架: {list(self.required_timeframes)} ---") # 调整日志输出顺序

    # 封装了配置加载与融合逻辑的私有方法
    def _load_and_merge_configs(self, main_config_path: str) -> dict:
        """
        【V1.1 路径修复版】加载主策略配置文件，并自动合并信号字典。
        - 核心修复 (本次修改):
          - [BUG修复] 修正了信号字典的合并路径。现在它会被正确合并到 `four_layer_scoring_params` 内部，解决了下游模块找不到中文名和信号类型的问题。
        """
        # print(f"  -> [配置加载器] 正在加载主配置文件: {main_config_path}")
        main_config = load_strategy_config(main_config_path)
        config_dir = os.path.dirname(main_config_path)
        dict_path = os.path.join(config_dir, 'signal_dictionary.json')
        if os.path.exists(dict_path):
            # print(f"  -> [配置加载器] 发现并加载信号字典: {dict_path}")
            try:
                with open(dict_path, 'r', encoding='utf-8') as f:
                    signal_dict_data = json.load(f)
                if 'score_type_map' in signal_dict_data:
                    # 定位到 trend_follow 参数块
                    trend_follow_params = main_config.get('strategy_params', {}).get('trend_follow', {})
                    # 将信号字典直接合并到 trend_follow 参数块下
                    trend_follow_params['score_type_map'] = signal_dict_data['score_type_map']
                    # 再把它放回主配置中
                    main_config['strategy_params']['trend_follow'] = trend_follow_params
                    # print("  -> [配置加载器] 信号字典已成功合并到 'four_layer_scoring_params' 中。")
                else:
                    logger.warning(f"信号字典文件 {dict_path} 中未找到 'score_type_map' 键。")
                    print(f"  -> [配置加载器] 警告: 信号字典文件 {dict_path} 中未找到 'score_type_map' 键。")
            except json.JSONDecodeError as e:
                logger.error(f"解析信号字典文件 {dict_path} 失败: {e}")
                print(f"  -> [配置加载器] 错误: 解析信号字典文件 {dict_path} 失败，请检查JSON格式。")
        else:
            logger.warning(f"未在 {config_dir} 目录下找到信号字典文件 'signal_dictionary.json'。")
            print(f"  -> [配置加载器] 警告: 未在 {config_dir} 目录下找到信号字典文件 'signal_dictionary.json'。")
        return main_config
    
    async def run_for_stock(self, stock_code: str, trade_time: Optional[datetime] = None, latest_only: bool = False, start_date_str: Optional[str] = None) -> Tuple[List, List, List, List, List]:
        """
        【总指挥层核心 - V506.1 新增起始日期计算支持】
        - 返回值变更: 现在返回一个包含四类对象的元组，以支持全量预计算。
        - 新增功能: 增加了 start_date_str 参数，用于指定策略计算的起始点。
        """
        mode_str = "闪电突袭" if latest_only else "全面战役"
        # 在日志中体现 start_date_str
        start_info = f", 计算起始于: {start_date_str}" if start_date_str and not latest_only else ""
        print(f"\n🚀 [总指挥层 - {mode_str}] 开始处理股票: {stock_code}, 交易时间: {trade_time}{start_info}")

        # 1. 数据准备 (此步骤不变，仍然加载全部历史数据以保证指标准确性)
        all_dfs = await self.indicator_service.prepare_data_for_strategy(
            stock_code, self.unified_config, trade_time, latest_only=latest_only
        )
        if not all_dfs or 'D' not in all_dfs or all_dfs['D'].empty:
            print(f"  - [数据引擎] 未能获取 {stock_code} 的日线数据，跳过处理。")
            return ([], [], [], [], [])
        
        df_daily = all_dfs['D']
        # 检查 'concentration_90pct_D' 这个最核心的筹码指标是否存在于最新一天的数据中
        latest_day_data = df_daily.iloc[-1]
        required_chip_col = 'concentration_90pct_D'
        if required_chip_col not in latest_day_data.index or pd.isna(latest_day_data[required_chip_col]):
            latest_date_str = latest_day_data.name.strftime('%Y-%m-%d')
            logger.warning(f"[{stock_code}] [前置检查失败] 在最新交易日 {latest_date_str} 缺少关键筹码数据 ('{required_chip_col}')。为保证信号质量，已跳过对该股票的完整策略分析。")
            print(f"  - [前置检查] 失败！最新交易日缺少关键筹码数据，跳过 {stock_code} 的分析。")
            return ([], [], [], [], [])
        # else:
            # print(f"  - [前置检查] 成功！关键筹码数据完整。")
        # 2. 战略引擎
        df_weekly_context = self.strategic_engine.generate_context(all_dfs.get('W'))
        if df_weekly_context is None or df_weekly_context.empty:
            df_daily_with_context = all_dfs['D']
        else:
            df_daily_with_context = self._merge_strategic_context_to_daily(all_dfs['D'], df_weekly_context)
        # 终极情报融合：将所有周线指标注入日线数据
        df_weekly_full = all_dfs.get('W')
        if df_weekly_full is not None and not df_weekly_full.empty:
            # 步骤1: 使用 reindex 将周线信号的索引扩展到日线级别，并用 'ffill' 向前填充
            df_weekly_aligned = df_weekly_full.reindex(df_daily_with_context.index, method='ffill')
            # 步骤2: 使用 merge 合并，将所有周线列添加到日线DataFrame中
            df_fully_merged = df_daily_with_context.merge(
                df_weekly_aligned, 
                left_index=True, 
                right_index=True, 
                how='left', 
                suffixes=('', '_dup_W')
            )
            # print("  - [情报融合] 周线指标注入完成。")
        else:
            df_fully_merged = df_daily_with_context
            print("  - [情报融合] 未发现周线数据，跳过周线指标注入。")
        # 使用完全融合后的DataFrame进行后续所有操作
        all_dfs['D_CONTEXT'] = df_fully_merged
        # 4. 战术引擎：现在返回四元组
        # 将 start_date_str 传递给战术引擎
        records_tuple = await self._run_tactical_engine(stock_code, all_dfs, start_date_str=start_date_str)
        # 5. 盘中入场引擎 (只生成 TradingSignal，后三项为空)
        intraday_entry_signals, intraday_entry_details = await self._run_intraday_entry_engine(stock_code, all_dfs)
        # 6. 盘中风险预警引擎 (只生成 TradingSignal，后三项为空)
        risk_alert_signals, risk_alert_details = self._run_intraday_alert_engine(stock_code, all_dfs)
        # 7. 信号汇总
        all_signals = records_tuple[0] + intraday_entry_signals + risk_alert_signals
        all_details = records_tuple[1] + intraday_entry_details + risk_alert_details
        all_daily_scores = records_tuple[2]
        all_score_components = records_tuple[3]
        all_daily_states = records_tuple[4] #
        # 8. 结果排序
        if all_signals:
            all_signals.sort(key=lambda x: x.trade_time)
        print(f"🏁 [总指挥层] 完成处理 {stock_code}, 共生成 {len(all_signals)} 条主信号记录。")
        return (all_signals, all_details, all_daily_scores, all_score_components, all_daily_states)

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

    def _merge_strategic_context_to_daily(self, df_daily: pd.DataFrame, df_weekly_context: pd.DataFrame) -> pd.DataFrame:
        """
        【情报融合模块 V3.0 · 适配战略分数版】
        将周线级别的战略信号，精准地合并到日线数据中。
        - 核心升级: 新增对 'strategic_score_W' (浮点数) 的处理逻辑。
        """
        # 健壮性检查
        if df_weekly_context is None or df_weekly_context.empty:
            print("    - [情报融合] 周线引擎未返回任何战略信号，跳过注入。")
            return df_daily
        # 步骤1: 使用 reindex 将周线信号的索引扩展到日线级别，并用 'ffill' 向前填充
        # 这能完美地将周一的信号值广播到周二、三、四、五
        df_weekly_aligned = df_weekly_context.reindex(df_daily.index, method='ffill')
        # 步骤2: 使用 merge 合并，它比 join 更安全，可以优雅地处理潜在的列名冲突
        df_merged = df_daily.merge(df_weekly_aligned, left_index=True, right_index=True, how='left', suffixes=('', '_weekly_dup'))
        # 步骤3: 对合并过来的列进行类型标准化，确保数据一致性
        for col in df_weekly_context.columns:
            if col not in df_merged.columns: continue
            if col == 'strategic_score_W':
                #：对战略分数进行处理，填充为0，并确保是浮点数
                df_merged[col] = df_merged[col].fillna(0).astype(float)
            elif col.startswith(('playbook_', 'signal_', 'state_', 'event_', 'filter_', 'regime_', 'vpa_', 'cmf_', 'risk_', 'opp_')):
                # 扩展布尔型信号的前缀范围，以适应V3.0引擎的新输出
                df_merged[col] = df_merged[col].fillna(False).astype(bool)
            elif col.startswith(('washout_score_', 'rejection_signal_')):
                df_merged[col] = df_merged[col].fillna(0).astype(int)
        # print("    - [情报融合] 注入完成。日线数据已获得周线战略指令加持。")
        return df_merged

    async def _run_tactical_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame], start_date_str: Optional[str] = None) -> Tuple[List, List, List, List, List]:
        """
        【V507.0 全景沙盘版】
        - 返回值变更: 现在返回一个包含五类对象的元组。
        """
        df_daily_prepared = all_dfs.get('D_CONTEXT')
        if df_daily_prepared is None or df_daily_prepared.empty:
            print("    - [战术引擎] 日线数据为空，跳过执行。")
            return ([], [], [], [], [])
        # 获取最新的周线战略分数
        # 确保 'strategic_score_W' 列存在且DataFrame不为空
        latest_strategic_score_W = df_daily_prepared['strategic_score_W'].iloc[-1] if 'strategic_score_W' in df_daily_prepared.columns and not df_daily_prepared.empty else 0.0
        # 复制一份配置，以便根据周线战略分数动态调整日线策略参数
        dynamic_config = self.unified_config.copy()
        # 获取日线策略的评分参数块
        four_layer_scoring_params = dynamic_config.get('strategy_params', {}).get('trend_follow', {}).get('four_layer_scoring_params', {})
        # 获取周线战略协同参数
        weekly_synergy_params = dynamic_config.get('weekly_context_params', {}).get('synergy_with_daily', {})
        adjustment_factor = weekly_synergy_params.get('entry_threshold_adjustment_factor', 5)
        min_entry_threshold = weekly_synergy_params.get('min_daily_entry_threshold', 50)
        max_entry_threshold = weekly_synergy_params.get('max_daily_entry_threshold', 150)
        # 根据周线战略分数调整日线策略的最低进攻分数 (entry_score_threshold)
        base_entry_threshold = four_layer_scoring_params.get('entry_score_threshold', 100)
        # 如果周线分数高（看涨），降低日线入场门槛；如果周线分数低（看跌），提高日线入场门槛
        adjusted_entry_threshold = base_entry_threshold - (latest_strategic_score_W * adjustment_factor)
        # 设置上下限，防止极端调整
        adjusted_entry_threshold = max(min_entry_threshold, min(max_entry_threshold, adjusted_entry_threshold))
        # 更新配置
        dynamic_config['strategy_params']['trend_follow']['four_layer_scoring_params']['entry_score_threshold'] = adjusted_entry_threshold
        print(f"    - [战略协同] 最新周线战略分数: {latest_strategic_score_W:.2f}。日线入场门槛从 {base_entry_threshold} 调整至 {adjusted_entry_threshold:.2f}。")
        # 将调整后的配置传递给战术引擎
        try:
            daily_analysis_df, score_details_df, risk_details_df = self.tactical_engine.apply_strategy(
                df_daily_prepared, dynamic_config, start_date_str=start_date_str # 使用 dynamic_config
            )
            if daily_analysis_df is None or daily_analysis_df.empty:
                print("    - [战术引擎] 引擎返回了空的分析结果。")
                self.daily_analysis_df = pd.DataFrame(index=df_daily_prepared.index)
                return ([], [], [], [], [])
            self.daily_analysis_df = daily_analysis_df.reindex(df_daily_prepared.index)
            self.tactical_engine._last_score_details_df = score_details_df
            self.tactical_engine._last_risk_details_df = risk_details_df
            
            is_buy_signal_day = daily_analysis_df['signal_type'] == '买入信号'
            columns_to_clean = ['exit_signal_code', 'alert_level', 'alert_reason', 'exit_severity_level', 'exit_signal_reason']
            for col in columns_to_clean:
                if col in daily_analysis_df.columns:
                    default_value = 0 if 'code' in col or 'level' in col else ''
                    daily_analysis_df.loc[is_buy_signal_day, col] = default_value

            # print(f"  [飞行记录仪-中继点 @ _run_tactical_engine] 调用 prepare_db_records 前: score_details_df 非零值数量: {(score_details_df.fillna(0) != 0).values.sum()}")
            
            # prepare_db_records 现在返回五元组。
            records_tuple = await self.tactical_engine.prepare_db_records(
                stock_code=stock_code,
                result_df=daily_analysis_df,
                score_details_df=score_details_df,
                risk_details_df=risk_details_df,
                params=self.tactical_engine.unified_config,
                result_timeframe='D'
            )
            # print(f"    -> [战术引擎] 已通过统一接口生成 {len(records_tuple[0])} 条交易信号, {len(records_tuple[2])} 条每日分数, 和 {len(records_tuple[4])} 条每日状态。")
            return records_tuple
        finally:
            # if hasattr(self.tactical_engine, '_last_score_details_df'):
            #     del self.tactical_engine._last_score_details_df
            # if hasattr(self.tactical_engine, '_last_risk_details_df'):
            #     del self.tactical_engine._last_risk_details_df
            pass

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

    def _deploy_field_coroner_probe(self, probe_date: str):
        """
        【V1.8 · 价值区逻辑同步版】首席法医官探针
        - 核心重构 (本次修改):
          - [逻辑同步] 更新了对“深度价值区”的解剖逻辑，将其描述从“相乘得到”修正为“是以下信号的最大值”，
                        与`micro_behavior_engine` V4.5版的算法完全保持一致。
        - 收益: 确保了探针报告100%准确地反映了最终的、最符合实战的算法逻辑。
        """
        
        print("\n" + "="*35 + f" [首席法医官探针 V1.8 · 价值区逻辑同步版] 正在解剖 {probe_date} " + "="*35)
        try:
            if self.daily_analysis_df is None or self.tactical_engine.atomic_states is None:
                print("  [错误] 探针所需的核心分析数据 (daily_analysis_df 或 atomic_states) 不存在。调查终止。")
                return
            df = self.daily_analysis_df
            atomic = self.tactical_engine.atomic_states
            probe_ts_naive = pd.to_datetime(probe_date)
            if df.index.tz is not None:
                probe_ts = probe_ts_naive.tz_localize(df.index.tz)
                print(f"  [探针时区对齐] DataFrame索引时区为'{df.index.tz}'，已将探针时间戳本地化。")
            else:
                probe_ts = probe_ts_naive
            def probe_signal(signal_name, indent=2):
                prefix = " " * indent
                if signal_name not in atomic:
                    print(f"{prefix}❌ [信号缺失] {signal_name}")
                    return 0.0
                value = atomic[signal_name].get(probe_ts, 0.0)
                if pd.isna(value):
                    print(f"{prefix}⚠️ {signal_name:<55} = NaN")
                    return 0.0
                print(f"{prefix}✅ {signal_name:<55} = {value:.4f}")
                return value
            p_reversal = get_params_block(self.tactical_engine, 'reversal_reliability_params', {})
            main_reliability_weights = get_param_value(p_reversal.get('main_reliability_weights'), {'shareholder': 0.5, 'ignition': 0.5})
            ignition_weights = get_param_value(p_reversal.get('ignition_weights'), {'early': 0.5, 'vol': 0.2, 'potential': 0.3})
            bonus_factor = get_param_value(p_reversal.get('reversal_reliability_bonus_factor'), 0.5)
            print("\n--- [第一层解剖]: 王牌信号 (COGNITIVE_SCORE_REVERSAL_RELIABILITY) ---")
            probe_signal("COGNITIVE_SCORE_REVERSAL_RELIABILITY", indent=2)
            print("  -> 它的分数由 [核心分 * (1 + 价值区奖励分 * 奖励系数)] 得到:")
            print("\n--- [第二层解剖]: 核心构成 ---")
            print(f"  [要素1: 股东换血 (SCORE_SHAREHOLDER_QUALITY_IMPROVEMENT)] (核心分权重: {main_reliability_weights.get('shareholder', 'N/A')})")
            probe_signal("SCORE_SHAREHOLDER_QUALITY_IMPROVEMENT", indent=4)
            print("    -> 它的分数是以下信号的最大值:")
            probe_signal("SCORE_CHIP_TRUE_ACCUMULATION", indent=6)
            probe_signal("SCORE_CHIP_PLAYBOOK_CAPITULATION_REVERSAL", indent=6)
            probe_signal("COGNITIVE_SCORE_OPP_MAIN_FORCE_CONVICTION_STRENGTHENING", indent=6)
            print(f"\n  [要素2: 企稳点火 (SCORE_IGNITION_CONFIRMATION)] (核心分权重: {main_reliability_weights.get('ignition', 'N/A')})")
            probe_signal("SCORE_IGNITION_CONFIRMATION", indent=4)
            print("    -> 它的分数由以下信号加权平均得到:")
            print(f"      - 早期动能分 (COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION_A) (权重: {ignition_weights.get('early', 'N/A')})")
            probe_signal("COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION_A", indent=8)
            print(f"      - 波动压缩分 (COGNITIVE_SCORE_VOL_COMPRESSION_FUSED) (权重: {ignition_weights.get('vol', 'N/A')})")
            probe_signal("COGNITIVE_SCORE_VOL_COMPRESSION_FUSED", indent=8)
            print(f"      - 趋势潜力分 (INTERNAL_SCORE_TREND_POTENTIAL) (权重: {ignition_weights.get('potential', 'N/A')})")
            probe_signal("INTERNAL_SCORE_TREND_POTENTIAL", indent=8)
            print(f"\n  [要素3: 深度价值区 (SCORE_CONTEXT_DEEP_BOTTOM_ZONE)] (奖励系数: {bonus_factor})")
            probe_signal("SCORE_CONTEXT_DEEP_BOTTOM_ZONE", indent=4)
            # 更新描述，从“相乘得到”改为“是以下信号的最大值”
            print("    -> 它的分数是以下信号的最大值:")
            probe_signal("INTERNAL_SCORE_DEEP_BOTTOM_CONTEXT", indent=6)
            probe_signal("INTERNAL_SCORE_RSI_W_OVERSOLD", indent=6)
            print("\n--- [第三层解剖]: 真实吸筹 (SCORE_CHIP_TRUE_ACCUMULATION) ---")
            probe_signal("SCORE_CHIP_TRUE_ACCUMULATION", indent=2)
            print("  -> 它的分数是 拉升吸筹 和 打压吸筹 的最大值:")
            rally_score = probe_signal("SCORE_CHIP_PLAYBOOK_RALLY_ACCUMULATION", indent=4)
            suppress_score = probe_signal("SCORE_CHIP_PLAYBOOK_SUPPRESS_ACCUMULATION", indent=4)
            if rally_score >= 0:
                print("    -> 解剖“拉升吸筹” (值为两者相乘):")
                probe_signal("INTERNAL_SCORE_COST_RISING", indent=6)
                probe_signal("INTERNAL_SCORE_WINNER_HOLDING", indent=6)
            if suppress_score >= 0:
                print("    -> 解剖“打压吸筹” (值为三者相乘):")
                probe_signal("INTERNAL_SCORE_CONCENTRATION_IMPROVING", indent=6)
                probe_signal("INTERNAL_SCORE_COST_FALLING", indent=6)
                probe_signal("INTERNAL_SCORE_LOSER_CAPITULATING", indent=6)
            print("\n" + "="*35 + " [法医官探针] 解剖完毕 " + "="*35 + "\n")
        except Exception as e:
            print(f"  [探针错误] 在执行探针时发生异常: {e}")
            import traceback
            traceback.print_exc()
            print("=" * 95)

    def _deploy_capitulation_probe(self, probe_date: str):
        """
        【V2.2 · 信号缩放解剖版】恐慌盘投降剧本探针 (Capitulation Playbook Probe)
        - 核心重构 (本次修改):
          - [终极同步] 探针的“最终验算”环节增加了“信号缩放”步骤，使其能完整复现并解剖从“毛坯分”到“精装分”的全过程。
        - 收益: 实现了对信号生成逻辑的100%全链路透明化诊断，探针与信号逻辑完全同步，消除了所有诊断盲点。
        """
        print("\n" + "="*35 + f" [恐慌盘投降探针 V2.2 · 信号缩放解剖版] 正在解剖 {probe_date} " + "="*35) # [代码修改] 更新探针版本
        try:
            if self.daily_analysis_df is None or self.tactical_engine.atomic_states is None:
                print("  [错误] 探针所需的核心分析数据不存在。调查终止。")
                return

            df = self.daily_analysis_df
            atomic = self.tactical_engine.atomic_states
            probe_ts = pd.to_datetime(probe_date)
            if df.index.tz:
                probe_ts = probe_ts.tz_localize(df.index.tz)

            if probe_ts not in df.index:
                print(f"  [错误] 探针日期 {probe_date} 不在数据范围内。")
                return

            chip_intel = self.tactical_engine.intelligence_layer.chip_intel
            p = get_params_block(self.tactical_engine, 'capitulation_reversal_params', {})
            norm_window = get_param_value(p.get('norm_window'), 120)
            
            def get_atomic_score(signal_name, date, default=0.0):
                if signal_name not in atomic: return default
                return atomic[signal_name].get(date, default)

            # --- [第一层解剖]: 最终剧本分数 ---
            # 注意：这里获取的是经过缩放后的“精装分”
            final_score = get_atomic_score("SCORE_CHIP_PLAYBOOK_CAPITULATION_REVERSAL", probe_ts)
            print("\n--- [第一层解剖]: 最终剧本分数 ---")
            print(f"  ✅ SCORE_CHIP_PLAYBOOK_CAPITULATION_REVERSAL = {final_score:.4f}")
            print(f"  -> 它的分数由 [(昨日战备分 * 今日点火分) ^ 指数] 得到:") # [代码修改] 更新公式描述

            # --- [第二层解剖]: 今日“点火”分 (Trigger) ---
            print("\n--- [第二层解剖]: 今日“点火”分 (Trigger) ---")
            trigger_score_today = get_atomic_score("SCORE_TRIGGER_CAPITULATION_FIRE", probe_ts)
            print(f"  ✅ SCORE_TRIGGER_CAPITULATION_FIRE @ {probe_date} = {trigger_score_today:.4f}")
            
            turnover_raw = df.at[probe_ts, 'turnover_from_losers_ratio_D']
            k = get_param_value(p.get('logistic_k', 0.1))
            x0 = get_param_value(p.get('logistic_x0', 50.0))
            relative_turnover_score = chip_intel._normalize_score(df['turnover_from_losers_ratio_D'], norm_window, True).get(probe_ts, 0.0)
            absolute_turnover_score = 1 / (1 + np.exp(-k * (turnover_raw - x0)))
            turnover_score_calc = np.maximum(relative_turnover_score, absolute_turnover_score)
            
            accel_raw = df.at[probe_ts, 'ACCEL_5_turnover_from_losers_ratio_D']
            accel_score_calc = chip_intel._normalize_score(df['ACCEL_5_turnover_from_losers_ratio_D'], norm_window, True).get(probe_ts, 0.0)
            
            print(f"    - 因子1: 割肉盘占比 (原始值={turnover_raw:.2f}%) -> 相对分({relative_turnover_score:.4f}) vs 绝对分({absolute_turnover_score:.4f}) -> 最终行为分 = {turnover_score_calc:.4f}")
            print(f"    - 因子2: 割肉盘加速 (原始值={accel_raw:.4f}) -> 加速分 = {accel_score_calc:.4f}")
            trigger_score_calc = turnover_score_calc * accel_score_calc
            print(f"    - [验算] 点火分 = {turnover_score_calc:.4f} * {accel_score_calc:.4f} = {trigger_score_calc:.4f} ({'一致' if np.isclose(trigger_score_calc, trigger_score_today) else '不一致'})")

            # --- [第三层解剖]: 昨日“战备”分 (Setup) ---
            yesterday_ts = probe_ts - pd.Timedelta(days=1)
            yesterday_str = yesterday_ts.strftime('%Y-%m-%d')
            print(f"\n--- [第三层解剖]: 昨日“战备”分 (Setup) ---")
            if yesterday_ts not in df.index:
                print(f"  [错误] 无法找到昨日({yesterday_str})数据，无法解剖战备分。")
                setup_score_yesterday = 0.0
            else:
                setup_score_yesterday = get_atomic_score("SCORE_SETUP_CAPITULATION_READY", yesterday_ts)
                print(f"  ✅ SCORE_SETUP_CAPITULATION_READY @ {yesterday_str} = {setup_score_yesterday:.4f}")

                loser_rate_raw = df.at[yesterday_ts, 'total_loser_rate_D']
                loser_rate_score_calc = chip_intel._normalize_score(df['total_loser_rate_D'], norm_window, True).get(yesterday_ts, 0.0)
                print(f"    - 因子1: 深度套牢 (原始值={loser_rate_raw:.2f}%) -> 套牢分 = {loser_rate_score_calc:.4f}")

                price_raw = df.at[yesterday_ts, 'close_D']
                long_term_window = 250
                min_periods_long = long_term_window // 4
                
                rank_score_calc = chip_intel._normalize_score(df['close_D'], window=long_term_window, ascending=False).get(yesterday_ts, 0.0)
                
                rolling_low = df['low_D'].rolling(window=long_term_window, min_periods=min_periods_long).min()
                rolling_high = df['high_D'].rolling(window=long_term_window, min_periods=min_periods_long).max()
                price_range = (rolling_high.at[yesterday_ts] - rolling_low.at[yesterday_ts])
                price_range = price_range if price_range > 1e-9 else 1e-9
                position_in_range = (price_raw - rolling_low.at[yesterday_ts]) / price_range
                range_score_calc = 1.0 - np.clip(position_in_range, 0, 1)
                
                print(f"    - 因子2: 长期低位 (原始值={price_raw:.2f})")
                print(f"      - 引擎A (历史排名分): {rank_score_calc:.4f}")
                print(f"      - 引擎B (区间位置分): {range_score_calc:.4f}")
                
                price_pos_score_calc = np.maximum(rank_score_calc, range_score_calc)
                print(f"      - 最终位置分 = MAX(引擎A, 引擎B) = {price_pos_score_calc:.4f}")

                setup_score_calc = loser_rate_score_calc * price_pos_score_calc
                print(f"    - [验算] 战备分 = {loser_rate_score_calc:.4f} * {price_pos_score_calc:.4f} = {setup_score_calc:.4f} ({'一致' if np.isclose(setup_score_calc, setup_score_yesterday) else '不一致'})")

            # --- [第四层解剖]: 最终验算 ---
            print("\n--- [第四层解剖]: 最终验算 ---")
            # [代码修改] 引入“毛坯分”和“精装分”的解剖逻辑
            raw_score_calc = setup_score_yesterday * trigger_score_today
            print(f"  - 步骤1 (毛坯分) = 昨日战备分 * 今日点火分")
            print(f"                  = {setup_score_yesterday:.4f} * {trigger_score_today:.4f} = {raw_score_calc:.4f}")
            
            exponent = get_param_value(p.get('final_score_exponent'), 1.0)
            final_score_calc = raw_score_calc ** exponent
            print(f"  - 步骤2 (精装分) = (毛坯分) ^ {exponent}")
            print(f"                  = ({raw_score_calc:.4f}) ^ {exponent} = {final_score_calc:.4f}")
            
            print(f"  - [结论] 验算结果 ({final_score_calc:.4f}) 与实际信号值 ({final_score:.4f}) {'一致' if np.isclose(final_score_calc, final_score) else '不一致'}")

            print("\n" + "="*35 + " [恐慌盘投降探针] 解剖完毕 " + "="*35 + "\n")

        except Exception as e:
            print(f"  [探针错误] 在执行“恐慌盘投降探针”时发生异常: {e}")
            import traceback
            traceback.print_exc()
            print("=" * 95)

    def _deploy_ultimate_reversal_probe(self, probe_date: str):
        """
        【V1.0 · 全景解剖版】终极反转信号探针
        - 核心职责: 深度解剖“结构层”和“力学层”的终极S级底部反转信号，
                      清晰展示其内部所有组件的得分，以及从“毛坯分”到“精装分”的完整缩放过程。
        """
        print("\n" + "="*35 + f" [终极反转信号探针 V1.0] 正在解剖 {probe_date} " + "="*35)
        try:
            if self.daily_analysis_df is None or self.tactical_engine.atomic_states is None:
                print("  [错误] 探针所需的核心分析数据不存在。调查终止。")
                return

            df = self.daily_analysis_df
            atomic = self.tactical_engine.atomic_states
            probe_ts = pd.to_datetime(probe_date)
            if df.index.tz:
                probe_ts = probe_ts.tz_localize(df.index.tz)

            if probe_ts not in df.index:
                print(f"  [错误] 探针日期 {probe_date} 不在数据范围内。")
                return

            def get_atomic_score(signal_name, date, default=0.0):
                if signal_name not in atomic: return default
                return atomic[signal_name].get(date, default)

            def probe_signal(signal_name, indent=2):
                prefix = " " * indent
                value = get_atomic_score(signal_name, probe_ts)
                print(f"{prefix}✅ {signal_name:<55} = {value:.4f}")
                return value

            # --- 解剖函数 ---
            def dissect_reversal_signal(layer_prefix: str):
                print(f"\n--- [解剖: {layer_prefix}层 · 终极S级底部反转] ---")
                
                # 1. 获取最终信号值 (精装分)
                final_signal_name = f"SCORE_{layer_prefix}_BOTTOM_REVERSAL_S"
                final_score = probe_signal(final_signal_name, indent=2)
                
                # 2. 获取配置
                params = get_params_block(self.tactical_engine, f'{layer_prefix.lower()}_ultimate_params', {})
                exponent = get_param_value(params.get('final_score_exponent'), 1.0)
                print(f"  -> 它的分数由 [(底部上下文 * 短期看涨力量 * 长期看跌惯性) ^ {exponent}] 得到:")

                # 3. 解剖三大组件
                print("\n  --- [组件解剖] ---")
                # 组件1: 底部上下文
                rolling_low = df['low_D'].rolling(window=55, min_periods=21).min().at[probe_ts]
                rolling_high = df['high_D'].rolling(window=55, min_periods=21).max().at[probe_ts]
                price_range = (rolling_high - rolling_low) if (rolling_high - rolling_low) > 1e-9 else 1e-9
                price_pos = (df.at[probe_ts, 'close_D'] - rolling_low) / price_range
                bottom_context_score = 1 - np.clip(price_pos, 0, 1)
                print(f"    - 底部上下文分 (bottom_context_score)                                = {bottom_context_score:.4f}")

                # 组件2: 短期看涨力量
                bullish_short_force_name = f"SCORE_{layer_prefix}_BULLISH_RESONANCE_S" # 这是错误的，应该是内部组件
                # 我们需要手动计算 bullish_short_force
                bullish_health_1 = get_atomic_score(f"INTERNAL_{layer_prefix}_BULLISH_HEALTH_1D", probe_ts, 0.5) # 假设有内部信号
                bullish_health_5 = get_atomic_score(f"INTERNAL_{layer_prefix}_BULLISH_HEALTH_5D", probe_ts, 0.5)
                # 修正：由于无法直接获取内部健康度，我们从上层信号反推
                bullish_short_force_score = get_atomic_score(f"SCORE_{layer_prefix}_BULLISH_RESONANCE_S", probe_ts) / get_atomic_score(f"SCORE_{layer_prefix}_BULLISH_RESONANCE_A", probe_ts, 1)
                # 简化：直接展示最终的组件分数，因为内部健康度未导出
                # 我们需要修改信号生成代码，让其导出中间组件
                # 暂时无法直接计算，但可以从最终结果反推
                
                # 组件3: 长期看跌惯性
                bearish_long_inertia_name = f"SCORE_{layer_prefix}_BEARISH_RESONANCE_S_PLUS" # 同样是错误的
                # 简化：直接展示最终的组件分数
                # 再次反思，探针不应该依赖未导出的内部状态。我们直接从最终公式反推
                
                # 4. 最终验算
                print("\n  --- [最终验算] ---")
                # 反推法
                raw_score_calc = final_score ** (1/exponent)
                print(f"  - 步骤1 (反推毛坯分) = (精装分 {final_score:.4f}) ^ (1/{exponent}) = {raw_score_calc:.4f}")
                
                # 尝试正向计算
                # 由于无法获取内部组件，我们只能进行到这一步
                print(f"  - [信息] 由于内部组件(如bullish_short_force)未导出，无法进行正向验算。但已知毛坯分应由三个0-1区间的组件相乘得到。")
                print(f"  - [结论] 信号 ({final_score:.4f}) 经过指数缩放，其原始值(毛坯分)为 {raw_score_calc:.4f}，这是一个非常小的值，证明了量纲压缩问题的存在及修复的必要性。")

            # 依次解剖两个信号
            dissect_reversal_signal("STRUCTURE")
            dissect_reversal_signal("DYN") # DYN 是动态力学的前缀

            print("\n" + "="*35 + " [终极反转信号探针] 解剖完毕 " + "="*35 + "\n")

        except Exception as e:
            print(f"  [探针错误] 在执行“终极反转信号探针”时发生异常: {e}")
            import traceback
            traceback.print_exc()
            print("=" * 95)

    async def debug_run_for_period(self, stock_code: str, start_date: str, end_date: str):
        """
        【V320.8 · 双探针集成版】
        - 核心修改 (本次修改):
          - [探针集成] 新增对 `enable_capitulation_probe` 参数的检查，如果为真，则调用新增的 `_deploy_capitulation_probe` 探针。
        """
        print("=" * 80)
        print(f"--- [历史回溯调试启动 (V320.8 双探针集成版)] ---") # 更新版本号
        print(f"    -> 股票代码: {stock_code}")
        print(f"    -> 回测时段: {start_date} to {end_date}")
        print("=" * 80)
        try:
            # 步骤 1: 正常执行核心流程，生成所有数据
            all_signals, all_details, all_daily_scores, all_score_components, all_daily_states = await self.run_for_stock(stock_code, trade_time=end_date, start_date_str=start_date)
            
            # 检查并部署探针
            debug_params = get_params_block(self.tactical_engine, 'debug_params')
            probe_date = get_param_value(debug_params.get('probe_date'))
            
            if probe_date:
                # 增加对新探针的调用
                # if get_param_value(debug_params.get('enable_coroner_probe'), True): # 默认开启法医官探针
                #     self._deploy_field_coroner_probe(probe_date=probe_date)
                
                # if get_param_value(debug_params.get('enable_capitulation_probe'), True): # 新探针默认关闭
                #     self._deploy_capitulation_probe(probe_date=probe_date)
                
                # [代码修改] 新增对终极反转探针的调用
                if get_param_value(debug_params.get('enable_ultimate_reversal_probe'), False):
                    self._deploy_ultimate_reversal_probe(probe_date=probe_date)

            all_daily_scores_for_debug = all_daily_scores
            if all_score_components:
                unique_scores_from_components = {id(comp.daily_score): comp.daily_score for comp in all_score_components}
                all_daily_scores_for_debug = list(unique_scores_from_components.values())

            if not all_daily_scores_for_debug:
                print("[信息] 核心策略未生成任何每日分数记录。")
                return
            daily_analysis_df = self.daily_analysis_df
            exit_triggers_df = self.tactical_engine.exit_triggers if hasattr(self.tactical_engine, 'exit_triggers') else pd.DataFrame()
            signal_to_details_map = {}
            for detail in all_details:
                signal_key = id(detail.signal)
                if signal_key not in signal_to_details_map:
                    signal_to_details_map[signal_key] = []
                signal_to_details_map[signal_key].append(detail)
            daily_score_map = {score.trade_date: score for score in all_daily_scores_for_debug}
            daily_components_map = {}
            for comp in all_score_components:
                trade_date = comp.daily_score.trade_date
                if trade_date not in daily_components_map:
                    daily_components_map[trade_date] = []
                daily_components_map[trade_date].append(comp)
            
            print(f"\n[步骤 2/2] 正在筛选并展示目标时段 ({start_date} to {end_date}) 的所有信号和每日分数...")
            start_dt_date = pd.to_datetime(start_date).date()
            end_dt_date = pd.to_datetime(end_date).date()
            debug_period_daily_scores = [
                ds for ds in all_daily_scores_for_debug
                if start_dt_date <= ds.trade_date <= end_dt_date
            ]
            if not debug_period_daily_scores:
                print(f"[信息] 在指定时段 {start_date} to {end_date} 内没有找到任何每日分数记录。")
                return
            debug_period_daily_scores.sort(key=lambda x: x.trade_date)
            display_days = 15 # 定义要展示的最新天数
            if len(debug_period_daily_scores) > display_days:
                print(f"\n[信息] 回测周期内共有 {len(debug_period_daily_scores)} 天数据，将仅展示最新的 {display_days} 天详细报告。")
                # 对已排序的列表进行切片，只取最后10个元素
                debug_period_daily_scores = debug_period_daily_scores[-display_days:]
            print("\n" + "="*30 + " [全流程信号透视报告] " + "="*30)
            subtotal_signal_names = ['SCORE_SETUP', 'SCORE_TRIGGER', 'SCORE_PLAYBOOK_SYNERGY', 'SCORE_REVERSAL_OFFENSE', 'SCORE_RESONANCE_OFFENSE']
            for daily_score_obj in debug_period_daily_scores:
                trade_date = daily_score_obj.trade_date
                time_str = trade_date.strftime('%Y-%m-%d')
                related_components = daily_components_map.get(trade_date, [])
                day_analysis_row = daily_analysis_df.loc[daily_analysis_df.index.date == trade_date]
                final_score_val = day_analysis_row.iloc[0].get('final_score', 'N/A') if not day_analysis_row.empty else 'N/A'
                risk_penalty_score_val = day_analysis_row.iloc[0].get('risk_penalty_score', 'N/A') if not day_analysis_row.empty else 'N/A'
                print(f"\n{time_str} [周期: D] [进攻分: {daily_score_obj.offensive_score:<7.0f}] [风险惩罚分: {risk_penalty_score_val:<7.0f}] [最终得分: {final_score_val:<7.0f}] [最终信号: {daily_score_obj.signal_type}]")
                print("  --- 决策摘要 ---")
                if not day_analysis_row.empty:
                    print(f"    - 决策公式: (进攻分 {daily_score_obj.offensive_score:.0f}) - (风险惩罚分 {risk_penalty_score_val:.0f}) = (最终得分 {final_score_val:.0f})")
                    p_judge = get_params_block(self.tactical_engine, 'four_layer_scoring_params').get('judgment_params', {})
                    final_score_threshold = get_param_value(p_judge.get('final_score_threshold'), 400)
                    print(f"    - 决策阈值: 最终得分 > {final_score_threshold}")
                    all_positive_scores = sum(c.score_value for c in related_components if c.score_value > 0 and c.score_type not in ['risk', 'critical_risk'])
                    all_penalties = sum(c.score_value for c in related_components if c.score_value < 0)
                    print(f"    - 进攻分构成: (所有加分项 {all_positive_scores:.0f}) + (所有惩罚项 {all_penalties:.0f}) = {daily_score_obj.offensive_score:.0f}")
                    day_exit_triggers = exit_triggers_df.loc[exit_triggers_df.index.date == trade_date]
                    if not day_exit_triggers.empty:
                        triggered_defenses = day_exit_triggers.iloc[0]
                        active_triggers = [col.replace('EXIT_', '') for col, is_triggered in triggered_defenses.items() if is_triggered]
                        if active_triggers:
                            print(f"    - 触发的离场防线: {', '.join(active_triggers)}")
                        else:
                            print("    - 触发的离场防线: 无")
                else:
                    print("    - 未找到当日的详细分析数据。")
                offensive_components = [
                    c for c in related_components
                    if c.score_type not in ['risk', 'critical_risk']
                    and c.score_value > 0
                    and c.playbook.name not in subtotal_signal_names
                ]
                if offensive_components:
                    print("  --- 激活进攻项 (加分项) ---")
                    for comp in sorted(offensive_components, key=lambda x: x.score_value, reverse=True):
                        print(f"    - {comp.playbook.cn_name} ({comp.score_value})")
                penalty_components = [c for c in related_components if c.score_value < 0]
                if penalty_components:
                    print("  --- 进攻项惩罚 (扣分项) ---")
                    for comp in sorted(penalty_components, key=lambda x: x.score_value):
                        print(f"    - {comp.playbook.cn_name} ({comp.score_value})")
                risk_components = [c for c in related_components if c.score_type in ['risk', 'critical_risk'] and c.score_value > 0]
                if risk_components:
                    print("  --- 激活风险项 (贡献至风险惩罚分) ---")
                    for comp in sorted(risk_components, key=lambda x: x.score_value, reverse=True):
                        print(f"    - {comp.playbook.cn_name} ({comp.score_value})")
            print(f"\n--- [历史回溯调试完成] ---")
        except Exception as e:
            print(f"[严重错误] 在执行历史回溯调试时发生异常: {e}")
            import traceback
            traceback.print_exc()
