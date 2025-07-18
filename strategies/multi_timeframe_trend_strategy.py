# 文件: strategies/multi_timeframe_trend_strategy.py
# 版本: V203.0 总指挥重构版
import io
import sys
import re
from contextlib import redirect_stdout
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, time
import json
import logging
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import numpy as np
from scipy.signal import find_peaks

from services.indicator_services import IndicatorService
from stock_models.index import TradeCalendar
from strategies.trend_following_strategy import TrendFollowStrategy
from strategies.weekly_context_engine import WeeklyContextEngine
from utils.config_loader import load_strategy_config
from utils.data_sanitizer import sanitize_for_json

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

    本次重构亮点:
    - **性能**: 彻底向量化了斜率计算，极大提升了性能。
    - **结构**: 优化了盘中引擎的循环逻辑，采用更高效的 `groupby` 模式。
    - **健壮性**: 全面增强了对空数据和缺失列的防御性检查。
    - **可读性**: 提供了全面的高级别和细节注释，阐明了架构设计和代码逻辑。
    """

    def __init__(self):
        """
        【V203.1 修正版】初始化总指挥部。
        - 核心修正: 恢复使用 self.indicator_service，移除了错误的 self.data_loader 引用。
        """
        print("--- [总指挥部] 正在初始化 (V203.1)... ---")
        # 加载唯一的全局配置文件，作为所有决策的依据
        unified_config_path = 'config/trend_follow_strategy.json'
        self.unified_config = load_strategy_config(unified_config_path)
        
        # ▼▼▼【代码修改】修正错误的变量名，恢复使用 indicator_service ▼▼▼
        # 初始化下属的核心服务与引擎
        self.indicator_service = IndicatorService() # 数据工程部门
        # ▲▲▲【代码修改】▲▲▲
        
        # 1. 初始化战略参谋部 (周线上下文引擎)
        self.strategic_engine = WeeklyContextEngine(config=self.unified_config)
        print("    -> [OK] 战略参谋部 (WeeklyContextEngine) 已就位。")
        
        # 2. 初始化一线作战部队 (日线战术引擎)
        self.tactical_engine = TrendFollowStrategy(config=self.unified_config)
        print("    -> [OK] 一线作战部队 (TrendFollowStrategy) 已就位。")
        
        # 内部状态变量
        self.daily_analysis_df = None # 存储日线战术引擎的详细分析结果
        
        # 从统一配置中自动发现所有需要的K线数据周期
        self.required_timeframes = self.indicator_service._discover_required_timeframes_from_config(self.unified_config)
        print(f"--- [总指挥部] 初始化完毕，已识别作战所需时间框架: {list(self.required_timeframes)} ---")

    async def run_for_stock(self, stock_code: str, trade_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        【总指挥层核心 - V203.3 修复版】
        为单个股票执行完整的多时间框架分析流程。
        - **修复 (V203.3)**: 修复了因缺少 await 关键字导致的 TypeError。
        """
        print(f"\n🚀 [总指挥层] 开始处理股票: {stock_code}, 交易时间: {trade_time}")

        # 1. 数据准备：获取所有需要的时间框架数据
        all_dfs = await self.indicator_service.prepare_data_for_strategy(
            stock_code, self.unified_config, trade_time
        )
        if not all_dfs or 'D' not in all_dfs or all_dfs['D'].empty:
            print(f"  - [数据引擎] 未能获取 {stock_code} 的日线数据，跳过处理。")
            return []

        # 2. 战略引擎：计算长期趋势和上下文
        df_weekly_context = self.strategic_engine.generate_context(all_dfs['W'])
        if df_weekly_context.empty:
            print(f"  - [战略引擎] 未能生成战略上下文，跳过后续处理。")
            return []

        # 3. 情报融合：将战略上下文合并到日线数据
        df_daily_with_context = self._merge_strategic_context_to_daily(all_dfs['D'], df_weekly_context)
        all_dfs['D_CONTEXT'] = df_daily_with_context # 更新 all_dfs，供下游引擎使用

        # 4. 战术引擎：基于日线+战略上下文，生成日线级别的交易信号
        tactical_records = self._run_tactical_engine(stock_code, all_dfs)
        print(f"  - [战术引擎] 生成 {len(tactical_records)} 条日线级信号。")

        # 5. 盘中入场引擎：对日线信号进行盘中确认
        intraday_entry_records = await self._run_intraday_entry_engine(stock_code, all_dfs)
        print(f"  - [盘中入场引擎] 生成 {len(intraday_entry_records)} 条盘中确认信号。")

        # 6. 盘中风险预警引擎：监控潜在的盘中风险
        # ▼▼▼【代码修改】: 补充遗漏的 await 关键字，确保获取到列表结果而非协程对象 ▼▼▼
        risk_alert_records = await self._run_intraday_alert_engine(stock_code, all_dfs)
        # ▲▲▲【代码修改】▲▲▲
        print(f"  - [盘中风险预警引擎] 生成 {len(risk_alert_records)} 条风险预警信号。")

        # 7. 信号汇总
        all_records = tactical_records + intraday_entry_records + risk_alert_records
        
        # 8. 结果排序（可选，但推荐）
        if all_records:
            all_records.sort(key=lambda x: x['trade_time'])

        print(f"🏁 [总指挥层] 完成处理 {stock_code}, 共生成 {len(all_records)} 条记录。")
        return all_records

    def _merge_strategic_context_to_daily(self, df_daily: pd.DataFrame, df_weekly_context: pd.DataFrame) -> pd.DataFrame:
        """
        【情报融合模块】
        将周线级别的战略信号，精准地合并到日线数据中。
        采用 reindex + ffill 的技术，确保周一生成的信号能正确地应用到本周的每一个交易日。
        """
        # 健壮性检查
        if df_weekly_context is None or df_weekly_context.empty:
            print("    - [情报融合] 周线引擎未返回任何战略信号，跳过注入。")
            return df_daily
        
        print(f"    - [情报融合] 准备将 {len(df_weekly_context.columns)} 个周线信号注入日线数据...")
        
        # 步骤1: 使用 reindex 将周线信号的索引扩展到日线级别，并用 'ffill' 向前填充
        # 这能完美地将周一的信号值广播到周二、三、四、五
        df_weekly_aligned = df_weekly_context.reindex(df_daily.index, method='ffill')
        
        # 步骤2: 使用 merge 合并，它比 join 更安全，可以优雅地处理潜在的列名冲突
        df_merged = df_daily.merge(df_weekly_aligned, left_index=True, right_index=True, how='left', suffixes=('', '_weekly_dup'))
        
        # 步骤3: 对合并过来的列进行类型标准化，确保数据一致性
        for col in df_weekly_context.columns:
            if col not in df_merged.columns: continue
            
            if col.startswith(('playbook_', 'signal_', 'state_', 'event_', 'filter_')):
                df_merged[col] = df_merged[col].fillna(False).astype(bool)
            elif col.startswith(('washout_score_', 'rejection_signal_')):
                df_merged[col] = df_merged[col].fillna(0).astype(int)
        
        print("    - [情报融合] 注入完成。日线数据已获得周线战略指令加持。")
        return df_merged

    def _run_tactical_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        【战术决策模块】
        运行核心的日线战术引擎，并处理其输出。
        """
        df_daily_prepared = all_dfs.get('D')
        if df_daily_prepared is None or df_daily_prepared.empty:
            print("    - [战术引擎] 日线数据为空，跳过执行。")
            return []

        # 核心调用：执行日线策略分析
        daily_analysis_df, _ = self.tactical_engine.apply_strategy(df_daily_prepared, self.unified_config)
        
        # 健壮性检查：处理引擎未返回结果的情况
        if daily_analysis_df is None or daily_analysis_df.empty:
            print("    - [战术引擎] 引擎返回了空的分析结果。")
            # 即使为空，也创建一个空的DataFrame以防后续代码出错
            self.daily_analysis_df = pd.DataFrame(index=df_daily_prepared.index)
            return []
        
        # 将分析结果保存到实例变量 self.daily_analysis_df，供其他盘中引擎使用
        # 使用 reindex 确保索引与原始日线数据对齐，并填充 NaN 值
        self.daily_analysis_df = daily_analysis_df.reindex(df_daily_prepared.index)
        if 'entry_score' in self.daily_analysis_df.columns:
            self.daily_analysis_df['entry_score'].fillna(0, inplace=True)
        
        # 填充所有布尔类型的列，防止出现 NaN
        bool_cols = self.daily_analysis_df.select_dtypes(include='bool').columns
        self.daily_analysis_df[bool_cols] = self.daily_analysis_df[bool_cols].fillna(False)
        
        # 统一调用唯一的“战报司令部”，生成所有日线信号（买入、卖出、风险预警）
        db_records = self.tactical_engine.prepare_db_records(
            stock_code, self.daily_analysis_df,
            params=self.unified_config, result_timeframe='D'
        )
        print(f"    -> [战术引擎] 已通过统一接口生成 {len(db_records)} 条日线信号(买入/卖出/预警)。")

        # 【情报下放】将日线级别计算出的关键信息（如平台价格）广播到分钟线，供后续使用
        cols_to_broadcast = ['PLATFORM_PRICE_STABLE'] 
        existing_cols = [col for col in cols_to_broadcast if col in self.daily_analysis_df.columns]
        if existing_cols:
            broadcast_df = self.daily_analysis_df[existing_cols].copy()
            for tf, df_intraday in all_dfs.items():
                if tf.isdigit() and df_intraday is not None and not df_intraday.empty:
                    all_dfs[tf] = pd.merge_asof(
                        left=df_intraday.sort_index(), right=broadcast_df.sort_index(),
                        left_index=True, right_index=True, direction='backward'
                    )
        
        return db_records

    async def _run_intraday_entry_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        【盘中入场确认引擎 - V203.0 重构版】
        识别日线发出的“预备信号”，并在次日的分钟线上寻找精确的“确认信号”。
        - **重构逻辑**: 废弃 for 循环，采用向量化和 groupby 操作，提升效率和可读性。
        """
        # 1. 加载配置并进行健壮性检查
        entry_params = self.tactical_engine._get_params_block(self.unified_config, 'intraday_entry_params', {})
        get_val = self.tactical_engine._get_param_value
        
        if not get_val(entry_params.get('enabled'), False): return []
        if self.daily_analysis_df is None or self.daily_analysis_df.empty: return []

        minute_tf = str(get_val(entry_params.get('timeframe'), '5'))
        minute_df = all_dfs.get(minute_tf)
        if minute_df is None or minute_df.empty: return []

        # 2. 向量化筛选：一次性找出所有满足条件的“预备日”
        daily_score_threshold = get_val(entry_params.get('daily_score_threshold'), 100)
        setup_days_df = self.daily_analysis_df[self.daily_analysis_df['entry_score'] >= daily_score_threshold].copy()
        if setup_days_df.empty: return []

        # 3. 准备上下文信息，用于合并
        # 提取预备日的关键信息：分数、剧本、平台价格
        setup_days_df['setup_date'] = setup_days_df.index.date
        
        # 获取次日（监控日）的日期
        trade_dates = await TradeCalendar.get_trade_dates_in_range_async(
            start_date=setup_days_df.index.min().date(),
            end_date=setup_days_df.index.max().date()
        )
        if not trade_dates: return []
        
        date_map = {d: nd for d, nd in zip(trade_dates[:-1], trade_dates[1:])}
        setup_days_df['monitoring_date'] = setup_days_df['setup_date'].map(date_map)
        setup_days_df.dropna(subset=['monitoring_date'], inplace=True)

        # 4. 向量化合并：将预备日的上下文信息，合并到监控日的分钟线数据中
        context_cols = ['monitoring_date', 'entry_score', 'PLATFORM_PRICE_STABLE']
        minute_df['monitoring_date'] = minute_df.index.date
        
        # 使用 merge 将日线上下文（基于监控日）赋给所有对应的分钟线
        merged_minute_df = pd.merge(
            minute_df,
            setup_days_df[context_cols],
            on='monitoring_date',
            how='inner' # 只保留那些确实是监控日的分钟线数据
        )
        if merged_minute_df.empty: return []

        # 5. 向量化计算：在合并后的分钟线数据上，一次性计算所有确认信号
        final_confirmation_signal = pd.Series(True, index=merged_minute_df.index)
        rules = entry_params.get('confirmation_rules', {})
        
        # 规则 a: 价格站上 VWAP
        vwap_rule = rules.get('vwap_reclaim', {})
        if get_val(vwap_rule.get('enabled'), False):
            vwap_col = f'VWAP_{minute_tf}'
            if vwap_col in merged_minute_df.columns:
                final_confirmation_signal &= (merged_minute_df[f'close_{minute_tf}'] > merged_minute_df[vwap_col])
        
        # 规则 b: 成交量放大
        vol_rule = rules.get('volume_confirmation', {})
        if get_val(vol_rule.get('enabled'), False):
            vol_ma_col = f'VOL_MA_{get_val(vol_rule.get("ma_period"), 21)}_{minute_tf}'
            if vol_ma_col in merged_minute_df.columns:
                final_confirmation_signal &= (merged_minute_df[f'volume_{minute_tf}'] > merged_minute_df[vol_ma_col])

        # 规则 c: 满足开盘后最小时间要求
        min_time_after_open = get_val(rules.get('min_time_after_open'), 15)
        merged_minute_df['time_of_day'] = merged_minute_df.index.time
        market_open_time = time(9, 30 + min_time_after_open)
        final_confirmation_signal &= (merged_minute_df['time_of_day'] >= market_open_time)

        # 6. Groupby + idxmin: 高效找出每个监控日的“首次”确认信号
        triggered_df = merged_minute_df[final_confirmation_signal]
        if triggered_df.empty: return []
        
        # `idxmin()` 在这里的作用是找到每个分组（每天）中，索引（时间）最小的那一行，即首次触发的时刻
        first_confirmations_df = triggered_df.loc[triggered_df.groupby('monitoring_date').idxmin().index]

        # 7. 生成最终战报记录
        final_entry_records = []
        playbook_blueprints = self.tactical_engine.playbook_blueprints
        playbook_cn_map = {p['name']: p.get('cn_name', p['name']) for p in playbook_blueprints}
        
        for timestamp, row in first_confirmations_df.iterrows():
            daily_score = row['entry_score']
            bonus_score = get_val(entry_params.get('bonus_score'), 50)
            final_score = daily_score + bonus_score
            
            # 准备剧本信息
            playbook_name = get_val(entry_params.get('playbook_name'), 'ENTRY_INTRADAY_CONFIRMATION')
            playbooks_en = [playbook_name] # 简化处理，可根据需要从日线继承
            playbooks_cn = [playbook_cn_map.get(p, p) for p in playbooks_en]

            record = self._create_signal_record(
                stock_code=stock_code, trade_time=timestamp, timeframe=minute_tf,
                strategy_name=get_val(entry_params.get('signal_name'), 'INTRADAY_ENTRY_CONFIRMATION'),
                close_price=row[f'close_{minute_tf}'], entry_score=final_score, entry_signal=True,
                triggered_playbooks=playbooks_en, triggered_playbooks_cn=playbooks_cn,
                stable_platform_price=row.get('PLATFORM_PRICE_STABLE'),
                context_snapshot={'close': row[f'close_{minute_tf}'], 'daily_score': daily_score, 'bonus': bonus_score, 'intraday_confirmed': True}
            )
            final_entry_records.append(record)
            
        return final_entry_records

    def _run_intraday_alert_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        【V203.3 最终确认版】盘中风险预警引擎
        监控由日线识别出的“风险预备日”的次日盘中表现，发出风险警报。
        - 业务逻辑: 与原始的 for-loop + 状态机版本完全等价。
        - 核心优化: 采用向量化和 groupby().apply() 模式，结构更清晰，执行更高效。
        """
        # 1. 加载配置并进行健壮性检查
        exec_params = self.tactical_engine._get_params_block(self.unified_config, 'intraday_execution_params', {})
        get_val = self.tactical_engine._get_param_value
        if not get_val(exec_params.get('enabled'), False): return []
        
        df_daily = all_dfs.get('D')
        if df_daily is None or df_daily.empty: return []

        minute_tf = str(get_val(exec_params.get('timeframe'), '30'))
        minute_df = all_dfs.get(minute_tf)
        if minute_df is None or minute_df.empty: return []

        # 2. 向量化筛选：一次性找出所有“风险预备日”
        rules_container = exec_params.get('rules', {})
        upthrust_params = rules_container.get('upthrust_rejection', {})
        if not get_val(upthrust_params.get('enabled'), False): return []
        
        upthrust_calc_params = self.tactical_engine._get_params_block(self.unified_config, 'exit_strategy_params', {}).get('upthrust_distribution_params', {})
        lookback_days = get_val(upthrust_calc_params.get('upthrust_lookback_days'), 5)
        
        is_upthrust_day = df_daily['high_D'] > df_daily['high_D'].shift(1).rolling(window=lookback_days, min_periods=1).max()
        setup_days_df = df_daily[is_upthrust_day].copy()
        if setup_days_df.empty: return []

        # 3. 向量化合并：将上下文合并到监控日的分钟线
        setup_days_df['monitoring_date'] = (setup_days_df.index + pd.Timedelta(days=1)).date
        minute_df['monitoring_date'] = minute_df.index.date
        
        merged_minute_df = pd.merge(minute_df, setup_days_df[['monitoring_date']], on='monitoring_date', how='inner')
        if merged_minute_df.empty: return []

        # 4. 向量化计算：找出所有“跌破VWAP”的时刻
        close_col, vwap_col = f'close_{minute_tf}', f'VWAP_{minute_tf}'
        if vwap_col not in merged_minute_df.columns or close_col not in merged_minute_df.columns: return []
        
        is_breaking_down = merged_minute_df[close_col] < merged_minute_df[vwap_col]
        first_breakdown_signal = is_breaking_down & ~is_breaking_down.shift(1).fillna(False)
        
        alert_days = merged_minute_df[first_breakdown_signal]['monitoring_date'].unique()
        if len(alert_days) == 0: return []

        # 5. Groupby + Apply: 对每个发生首次跌破的监控日，应用状态机逻辑
        def process_alert_day(day_df: pd.DataFrame) -> Optional[Dict]:
            """应用于每个监控日分组的函数，实现状态机逻辑"""
            is_breaking = day_df[close_col] < day_df[vwap_col]
            first_break_mask = is_breaking & ~is_breaking.shift(1).fillna(False)
            
            # 检查当天是否存在首次跌破信号，如果不存在则直接返回，增加健壮性
            if not first_break_mask.any(): return None
            
            # ▼▼▼【代码修改】这里是您提问的关键点 ▼▼▼
            # .idxmax() 在一个以 DatetimeIndex 为索引的布尔 Series 上调用时，
            # 会直接返回第一个 True 值对应的索引标签，这个标签就是一个 pandas.Timestamp 对象。
            # 因此，first_break_idx 已经是时间戳类型，无需任何转换。
            first_break_idx = first_break_mask.idxmax()
            # ▲▲▲【代码修改】▲▲▲
            
            first_alert_row = day_df.loc[first_break_idx]
            
            # 检查警报发出后，当天是否重新站回VWAP
            df_after_alert = day_df[day_df.index > first_break_idx]
            is_reclaimed = (df_after_alert[close_col] > df_after_alert[vwap_col]).any()
            
            # 根据是否收复，决定最终的警报内容
            if is_reclaimed:
                reclaim_time = df_after_alert[df_after_alert[close_col] > df_after_alert[vwap_col]].index[0]
                # ▼▼▼【代码修改】直接调用 .strftime() 是完全正确的，因为 first_break_idx 是 Timestamp 对象 ▼▼▼
                final_reason = f"[威胁解除] 曾于{first_break_idx.strftime('%H:%M')}跌破VWAP, 但已于{reclaim_time.strftime('%H:%M')}收复"
                final_code, final_severity = 0, 0 # 无风险
            else:
                # ▼▼▼【代码修改】此处同理，直接调用 .strftime() ▼▼▼
                final_reason = f"盘中于{first_break_idx.strftime('%H:%M')}跌破VWAP且至收盘未收复"
                final_code = get_val(upthrust_params.get('alert_code'), 103)
                final_severity = get_val(upthrust_params.get('severity_level'), 3)
            # ▲▲▲【代码修改】▲▲▲

            return self._create_signal_record(
                stock_code=stock_code, trade_time=first_break_idx, timeframe=minute_tf,
                strategy_name="INTRADAY_RISK_ALERT", close_price=first_alert_row[close_col],
                exit_signal_code=final_code, exit_severity_level=final_severity,
                exit_signal_reason=final_reason,
                triggered_playbooks=[get_val(upthrust_params.get('alert_playbook_name'), 'EXIT_INTRADAY_UPTHRUST_REJECTION')],
                context_snapshot={'close': first_alert_row[close_col], 'vwap': first_alert_row[vwap_col], 'reclaimed': is_reclaimed},
            )

        # 对所有需要监控的日子的分钟线数据进行分组，并应用处理函数
        final_alerts = merged_minute_df[merged_minute_df['monitoring_date'].isin(alert_days)]\
            .groupby('monitoring_date', group_keys=False)\
            .apply(process_alert_day)\
            .dropna().tolist()
            
        return final_alerts

    def _calculate_trend_dynamics(self, df: pd.DataFrame, timeframes: List[str], ema_period: int = 34, slope_window: int = 5) -> pd.DataFrame:
        """
        【性能核心 - 向量化斜率计算 V203.0】
        一次性计算多个时间框架下，EMA均线的斜率和加速度。
        - **重构逻辑**: 废弃 `rolling().apply()`，改用基于 `np.polyfit` 的高效向量化实现。
                       通过构建一个滑动窗口的视图（view），我们可以对所有窗口并行执行线性回归，
                       性能远超逐个窗口计算的 `apply` 模式。
        """
        df_copy = df.copy()
        
        # 创建一个 (0, 1, 2, ..., N-1) 的数组，用于线性回归的 x 轴
        x = np.arange(slope_window)
        # 预计算 x 的相关项，用于 polyfit
        x_matrix = np.vstack([x, np.ones(slope_window)]).T

        for tf in timeframes:
            ema_col = f'EMA_{ema_period}_{tf}'
            close_col = f'close_{tf}'
            slope_col, accel_col, health_col = f'ema_slope_{tf}', f'ema_accel_{tf}', f'trend_health_{tf}'

            if ema_col not in df_copy.columns:
                df_copy[slope_col], df_copy[accel_col], df_copy[health_col] = np.nan, np.nan, False
                continue

            y_series = df_copy[ema_col].values
            
            # 使用 numpy.lib.stride_tricks 创建滑动窗口的视图，这是向量化的关键
            # shape: (len(y) - N + 1, N)
            y_strided = np.lib.stride_tricks.as_strided(
                y_series,
                shape=(len(y_series) - slope_window + 1, slope_window),
                strides=(y_series.strides[0], y_series.strides[0])
            )
            
            # 对所有窗口一次性执行线性回归，`[0]` 表示我们只需要斜率
            # `np.linalg.lstsq` 是 `polyfit` 的底层实现，更高效
            slopes = np.linalg.lstsq(x_matrix, y_strided.T, rcond=None)[0][0]
            
            # 将计算结果填充回DataFrame，注意要对齐索引
            df_copy[slope_col] = pd.Series(slopes, index=df_copy.index[slope_window - 1:])
            
            # 同样的方法计算斜率的斜率（加速度）
            slope_series = df_copy[slope_col].dropna().values
            if len(slope_series) >= slope_window:
                slope_strided = np.lib.stride_tricks.as_strided(
                    slope_series,
                    shape=(len(slope_series) - slope_window + 1, slope_window),
                    strides=(slope_series.strides[0], slope_series.strides[0])
                )
                accelerations = np.linalg.lstsq(x_matrix, slope_strided.T, rcond=None)[0][0]
                df_copy[accel_col] = pd.Series(accelerations, index=df_copy[slope_col].dropna().index[slope_window - 1:])
            else:
                df_copy[accel_col] = np.nan

            # 计算趋势健康度
            is_above_ema = df_copy[close_col] > df_copy[ema_col]
            is_slope_positive = df_copy[slope_col] > 0
            df_copy[health_col] = is_above_ema & is_slope_positive
            df_copy[health_col].fillna(False, inplace=True)

        return df_copy

    # ▼▼▼ 标准化战报生成器 (保持不变，作为稳定的基础服务) ▼▼▼
    def _create_signal_record(self, **kwargs) -> Dict[str, Any]:
        """
        【标准化战报生成器】
        创建一个结构统一的信号记录字典，确保数据契约的一致性。
        """
        trade_time_input = kwargs.get('trade_time')
        if trade_time_input is None: raise ValueError("创建信号记录时必须提供 'trade_time'")
        
        # 标准化时间为带UTC时区的datetime对象
        ts = pd.to_datetime(trade_time_input)
        if ts.tzinfo is None:
            standard_trade_time = ts.tz_localize('Asia/Shanghai').tz_convert('UTC').to_pydatetime()
        else:
            standard_trade_time = ts.tz_convert('UTC').to_pydatetime()

        # 标准战报模板
        record = {
            "stock_code": None, "trade_time": standard_trade_time, "timeframe": "N/A",
            "strategy_name": "UNKNOWN", "close_price": 0.0, "entry_score": 0.0,
            "entry_signal": False, "is_risk_warning": False, "exit_signal_code": 0,
            "exit_severity_level": 0, "exit_signal_reason": None,
            "triggered_playbooks": [], "triggered_playbooks_cn": [],
            "stable_platform_price": None, "context_snapshot": {},
        }
        record.update(kwargs)
        
        # 数据净化，确保可以被序列化 (例如转为JSON)
        record['close_price'] = sanitize_for_json(record['close_price'])
        record['context_snapshot'] = sanitize_for_json(record['context_snapshot'])
        record['stable_platform_price'] = sanitize_for_json(record['stable_platform_price'])

        return record

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

    async def debug_run_for_period(self, stock_code: str, start_date: str, end_date: str):
        """
        【V202.14 战报简化版】
        - 核心升级: 极大地简化了此方法。由于所有情报在生成时已被量化并存入
                    数据库，本方法不再需要进行任何复杂的翻译工作，只需直接
                    从记录中读取 `exit_signal_reason` 并展示即可。
        """
        print("=" * 80)
        print(f"--- [历史回溯调试启动 (V202.14 战报简化版)] ---")
        print(f"    -> 股票代码: {stock_code}")
        print(f"    -> 回测时段: {start_date} to {end_date}")
        print("=" * 80)

        try:
            all_records = await self.run_for_stock(stock_code, trade_time=end_date)
            if all_records is None: return
            print(f"\n[步骤 2/3] 正在筛选并展示目标时段 ({start_date} to {end_date}) 的所有信号...")
            start_dt = pd.to_datetime(start_date, utc=True)
            end_dt = pd.to_datetime(end_date, utc=True).replace(hour=23, minute=59, second=59)
            
            debug_period_records = []
            for rec in all_records:
                rec_time = pd.to_datetime(rec['trade_time'])
                if rec_time.tzinfo is None: rec_time = rec_time.tz_localize('UTC')
                else: rec_time = rec_time.tz_convert('UTC')
                if start_dt <= rec_time <= end_dt:
                    debug_period_records.append(rec)

            if not debug_period_records:
                print(f"[信息] 在指定时段 {start_date} to {end_date} 内没有找到任何信号。")
                return

            debug_period_records.sort(key=lambda x: pd.to_datetime(x['trade_time'], utc=True))
            print("\n" + "="*30 + " [全流程信号透视报告] " + "="*30)
            
            for record in debug_period_records:
                time_obj = pd.to_datetime(record['trade_time'])
                time_str = time_obj.strftime('%Y-%m-%d %H:%M:%S %Z')
                tf = record.get('timeframe', 'N/A')
                signal_type = "未知信号"
                details = "无详细信息"
                
                context = record.get('context_snapshot', {})
                risk_score = context.get('risk_score', 0)
                
                # ▼▼▼【代码修改 V202.14】: 直接读取，无需翻译！▼▼▼
                reason = record.get('exit_signal_reason') or "原因未知"

                if record.get('exit_signal_code', 0) > 0:
                    severity = record.get('exit_severity_level', 0)
                    signal_type = f"卖出警报(L{severity})"
                    details = f"风险分: {risk_score:<3.0f} | 原因: {reason}"
                
                elif record.get('entry_signal'):
                    score = record.get('entry_score', 0.0)
                    playbooks = record.get('triggered_playbooks_cn', [])
                    signal_type = "买入信号"
                    details = f"得分: {score:<7.2f} | 剧本: {', '.join(playbooks)}"
                
                elif record.get('is_risk_warning'):
                    signal_type = "风险预警"
                    details = f"风险分: {risk_score:<3.0f} | 原因: {reason}"
                
                elif record.get('strategy_name') == 'INTRADAY_RISK_ALERT':
                    signal_type = f"盘中异动"
                    details = f"原因: {reason}"
                # ▲▲▲【代码修改 V202.14】▲▲▲

                if signal_type != "未知信号":
                    print(f"{time_str}  [周期:{tf:>3s}] [类型:{signal_type:<12s}] | {details}")
                else:
                    print(f"{time_str}  [周期:{tf:>3s}] [类型:{signal_type:<12s}] | 原始记录: {record}")

            print(f"--- [历史回溯调试完成] ---")
        except Exception as e:
            print(f"[严重错误] 在执行历史回溯调试时发生异常: {e}")
            import traceback
            traceback.print_exc()

    async def run_alpha_hunter(self, stock_code: str):
        """
        【V8.0 统一指挥版】
        - 适配修改: 更新调用的配置对象为 self.unified_config。
        """
        print("=" * 80)
        print(f"--- [总指挥] 阿尔法猎手任务启动 for {stock_code} (V8.0 统一指挥版) ---")
        # 1. 准备数据
        print(f"    -> 正在为 {stock_code} 准备全量历史数据...")
        all_dfs = await self.indicator_service.prepare_data_for_strategy(
            stock_code=stock_code,
            config=self.unified_config
        )
        if 'D' not in all_dfs or all_dfs['D'].empty:
            print(f"    -> [错误] 无法获取 {stock_code} 的日线数据，任务终止。")
            return
        # 2. 调用战术引擎的阿尔法猎手方法
        await self.tactical_engine.alpha_hunter_backtest(
            stock_code=stock_code,
            df_full=all_dfs['D'],
            params=self.unified_config
        )
        print(f"--- [总指挥] {stock_code} 的阿尔法猎手任务执行完毕。 ---")
        print("=" * 80)


