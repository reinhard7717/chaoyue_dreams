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

    async def run_for_stock(self, stock_code: str, trade_time: Optional[datetime] = None, latest_only: bool = False) -> List[Dict[str, Any]]:
        """
        【总指挥层核心 - V205.0 战略重构版】
        - 核心升级: 新增 latest_only 参数。当为 True 时，将指令传递给数据服务，
                    从源头上只加载少量近期数据，实现真正的“闪电突袭”。
        """
        mode_str = "闪电突袭" if latest_only else "全面战役"
        print(f"\n🚀 [总指挥层 - {mode_str}] 开始处理股票: {stock_code}, 交易时间: {trade_time}")

        # 1. 数据准备：将 latest_only 指令传递给数据引擎！
        all_dfs = await self.indicator_service.prepare_data_for_strategy(
            stock_code, self.unified_config, trade_time, latest_only=latest_only
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
        risk_alert_records = self._run_intraday_alert_engine(stock_code, all_dfs)
        print(f"  - [盘中风险预警引擎] 生成 {len(risk_alert_records)} 条风险预警信号。")

        # 7. 信号汇总
        all_records = tactical_records + intraday_entry_records + risk_alert_records
        
        # 8. 结果排序（可选，但推荐）
        if all_records:
            all_records.sort(key=lambda x: x['trade_time'])

        print(f"🏁 [总指挥层] 完成处理 {stock_code}, 共生成 {len(all_records)} 条记录。")
        return all_records

    async def run_for_latest_signal(self, stock_code: str, trade_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        【V204.0 闪电突袭模式】
        为日常任务设计的轻量化、高性能分析模式。
        - 作战流程:
          1. 依然调用“全面战役”模式，获取所有历史信号。
          2. 【核心优化】在获取所有信号后，只筛选出【最后一个交易日】的信号返回。
        - 收益: 极大减少了下游（数据库、状态更新）的处理负担，显著提升了日常任务的执行效率。
        """
        print(f"\n⚡️ [总指挥层] 接到“闪电突袭”指令，正在以高效模式处理: {stock_code}")
        
        # 直接调用“全面战役”引擎，但命令它以“闪电模式”运行！
        all_records = await self.run_for_stock(stock_code, trade_time, latest_only=True)
        
        if not all_records:
            print(f"  - [闪电突袭] 未发现任何信号，任务完成。")
            return []
            
        # 由于数据层已经做了优化，这里理论上只会返回近期的少量信号，但为保险起见，仍然执行筛选
        latest_date = max(rec['trade_time'].date() for rec in all_records)
        latest_records = [rec for rec in all_records if rec['trade_time'].date() == latest_date]
        
        print(f"🏁 [总指挥层-闪电突袭] 高效模式处理完毕, 共生成 {len(latest_records)} 条最新信号。")
        return latest_records

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

    def _deploy_field_coroner_probe(self, df: pd.DataFrame, probe_date: str, score_details: pd.DataFrame, risk_details: pd.DataFrame, **kwargs):
        """
        【首席法医官 V2.3：证据清单修正版】
        - 核心修正: 根据实际的DataFrame列名清单，全面重写了 rule_to_evidence_mapping。
                    确保法医官能够根据正确的标签，在军火库中找到所有需要的证据。
        - 新增逻辑:
          1. 所有列名都精确匹配了实际的命名约定（如 `_D` 后缀）。
          2. 对于不存在的“概念性”指标，暂时移除或替换为可观察的基础指标。
          3. 调整了部分指标的参数以匹配实际情况 (如 RSI_13_D)。
        """
        print("\n    ========================= [战地验尸总署-探针报告 V2.3] =========================")
        
        # ▼▼▼【代码修改 V2.3】: 全面修正“法医证据清单”以匹配实际列名 ▼▼▼
        rule_to_evidence_mapping = {
            # --- 风险规则的证据清单 ---
            # 注意：主力资金的5日/10日合计值可能需要数据工程层计算并输出，这里暂时使用单日值作为替代证据
            'risk_RISK_CAPITAL_STRUCT_MAIN_FORCE_DISTRIBUTING': ['main_force_net_inflow_amount_D', 'retail_net_inflow_volume_D'],
            'risk_STRUCTURE_TOPPING_DANGER_S': ['close_D', 'high_D', 'MACDh_13_34_8_D', 'RSI_13_D', 'BIAS_21_D', 'BIAS_55_D'],
            
            # --- 进攻规则的证据清单 ---
            # 注意：3日合计值和价格变动也需要数据工程层计算，暂时使用单日值
            'CAPITAL_DIVERGENCE_REVERSAL': ['main_force_net_inflow_amount_D', 'close_D', 'prev_20d_close_D'],
            'CHIP_HEALTH_EXCELLENT': ['chip_health_score_D', 'winner_profit_margin_D'],
            # 'platform_stability_score' 是概念分数，替换为可观察的基础指标
            'PLATFORM_STATE_STABLE_FORMED': ['peak_cost_slope_5d_D', 'peak_stability_D', 'close_D', 'EMA_55_D'],
            'CHIP_STATE_HIGHLY_CONCENTRATED': ['concentration_90pct_D', 'peak_stability_D'],
            # 'dyn_trend_score' 是概念分数，替换为构成它的核心斜率和加速度指标
            'DYN_TREND_HEALTHY_ACCELERATING': ['SLOPE_55_EMA_55_D', 'ACCEL_55_EMA_55_D', 'SLOPE_13_EMA_13_D'],
            # 'high_20d' 不存在，替换为与前20日收盘价的比较
            'CONTEXT_STRONG_BREAKOUT_RALLY': ['close_D', 'prev_20d_close_D', 'volume_D', 'VOL_MA_21_D', 'pct_change_D']
        }
        # ▲▲▲【代码修改 V2.3】▲▲▲

        try:
            probe_dt = pd.to_datetime(probe_date).date()
            probe_row = df.loc[df.index.date == probe_dt].iloc[0]
            # 修正：从 probe_row 中获取 stock_code，而不是依赖外部传入
            stock_code = probe_row.get('stock_code', 'N/A')
            if stock_code == 'N/A' and 'stock_id_D' in probe_row:
                 stock_code = probe_row['stock_id_D']
            print(f"      [验尸目标]: 股票代码 {stock_code} @ {probe_date}")
        except (IndexError, KeyError):
            print(f"      [错误] 未能在主数据流中找到目标日期 {probe_date} 的记录。")
            return

        def transform_wide_to_long(details_df: pd.DataFrame, target_date_str: str) -> pd.DataFrame:
            if details_df is None or details_df.empty:
                return pd.DataFrame(columns=['rule', 'score'])
            target_date = pd.to_datetime(target_date_str).date()
            if isinstance(details_df.index, pd.DatetimeIndex):
                day_details = details_df[details_df.index.date == target_date]
            else:
                print(f"      [警告] 详情案卷的索引不是日期格式，无法进行验尸。")
                return pd.DataFrame(columns=['rule', 'score'])
            if day_details.empty:
                return pd.DataFrame(columns=['rule', 'score'])
            long_df = day_details.melt(var_name='rule', value_name='score')
            long_df = long_df[long_df['score'] != 0].reset_index(drop=True)
            return long_df
        
        risk_rules_long = transform_wide_to_long(risk_details, probe_date)
        score_rules_long = transform_wide_to_long(score_details, probe_date)

        # --- 风险验尸科 ---
        print("  --- [风险验尸科 V297.0] 开始解剖风险成因 (协议已同步) ---")
        if not risk_rules_long.empty:
            print(f"  [目标日期 {probe_date} 风险详情]:")
            print(f"    -> 当日总风险分: {risk_rules_long['score'].sum():.2f}")
            print(f"    -> 风险构成:")
            for _, rule_row in risk_rules_long.iterrows():
                rule_name = rule_row['rule']
                rule_score = rule_row['score']
                print(f"      - {rule_name}: {rule_score:.2f} 分")
                
                evidence_cols = rule_to_evidence_mapping.get(rule_name, [])
                if not evidence_cols:
                    evidence_cols = rule_to_evidence_mapping.get(rule_name.replace('risk_', ''), [])

                if evidence_cols:
                    print(f"        -> [法医证据]:")
                    for col in evidence_cols:
                        try:
                            value = probe_row[col]
                            # 修正：对 pct_change_D 进行特殊格式化
                            if col == 'pct_change_D':
                                print(f"          - {col}: {value:.2%}")
                            elif isinstance(value, (int, float)):
                                print(f"          - {col}: {value:.2f}")
                            else:
                                print(f"          - {col}: {value}")
                        except KeyError:
                            print(f"          - [警告] 证据列 '{col}' 不存在!")
        else:
            print(f"    -> [信息] 在目标日期 {probe_date} 未发现任何风险信号。")

        # --- 进攻分验尸科 ---
        print(f"      --- [进攻分验尸科 V300.0] 开始解剖得分构成 (已接收全套案情卷宗) ---")
        if not score_rules_long.empty:
            print(f"      [目标日期 {probe_date} 得分详情]:")
            print(f"        -> 当日总得分: {score_rules_long['score'].sum():.2f}")
            print(f"        -> 得分构成:")
            for _, rule_row in score_rules_long.iterrows():
                rule_name = rule_row['rule']
                rule_score = rule_row['score']
                print(f"          - {rule_name}: {rule_score:.2f} 分")

                evidence_cols = rule_to_evidence_mapping.get(rule_name, [])
                if evidence_cols:
                    print(f"            -> [法医证据]:")
                    for col in evidence_cols:
                        try:
                            value = probe_row[col]
                            if col == 'pct_change_D':
                                print(f"              - {col}: {value:.2%}")
                            elif isinstance(value, (int, float)):
                                print(f"              - {col}: {value:.2f}")
                            else:
                                print(f"              - {col}: {value}")
                        except KeyError:
                            print(f"              - [警告] 证据列 '{col}' 不存在!")
        else:
            print(f"    -> [信息] 在目标日期 {probe_date} 未发现任何进攻得分信号。")
        
        print("    ============================== [验尸报告结束] ==============================")

    def _run_tactical_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        【V301.1 总司令部验尸版】
        - 核心重构: 将“临时情报中心”的建立与销毁职责，完全上移至本总司令部。
        - 新增功能: 在获取所有情报后，由总司令部直接部署“首席法医官”进行验尸。
        """
        # 关键修正：验尸官现在直接使用总司令部持有的、最完整的 df_daily_prepared
        df_daily_prepared = all_dfs.get('D_CONTEXT') # 使用融合了周线上下文的日线数据
        if df_daily_prepared is None or df_daily_prepared.empty:
            print("    - [战术引擎] 日线数据为空，跳过执行。")
            return []

        try:
            daily_analysis_df, score_details_df, risk_details_df = self.tactical_engine.apply_strategy(
                df_daily_prepared, self.unified_config
            )
            
            if daily_analysis_df is None or daily_analysis_df.empty:
                print("    - [战术引擎] 引擎返回了空的分析结果。")
                self.daily_analysis_df = pd.DataFrame(index=df_daily_prepared.index)
                return []

            # ▼▼▼【代码修改 V2.0】: 总司令部直接部署验尸官！▼▼▼
            debug_params = self.tactical_engine._get_params_block('debug_params')
            probe_date = self.tactical_engine._get_param_value(debug_params.get('probe_date'))
            
            if probe_date:
                print(f"    --- [总司令部] 接到密令！正在对 {probe_date} 的战况进行深度解剖... ---")
                # 调用自己内部的验尸官，并移交所有最原始的案情卷宗
                self._deploy_field_coroner_probe(
                    df=df_daily_prepared, # 传递最完整的、带有上下文的日线数据
                    probe_date=probe_date,
                    score_details=score_details_df,
                    risk_details=risk_details_df
                )
            # ▲▲▲【代码修改 V2.0】▲▲▲

            self.tactical_engine._last_score_details_df = score_details_df
            self.tactical_engine._last_risk_details_df = risk_details_df
            print("    -> [总司令部] 已完成现场归档，所有下属单位可访问。")

            self.daily_analysis_df = daily_analysis_df.reindex(df_daily_prepared.index)
            
            db_records = self.tactical_engine.prepare_db_records(
                stock_code=stock_code,
                result_df=daily_analysis_df,        # 修正: 使用战术引擎返回的 `daily_analysis_df`
                score_details_df=score_details_df,
                risk_details_df=risk_details_df,
                params=self.tactical_engine.unified_config,
                result_timeframe='D'
            )
            print(f"    -> [战术引擎] 已通过统一接口生成 {len(db_records)} 条日线信号(买入/卖出/预警)。")

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

        finally:
            print("    -> [总司令部] 正在执行“阅后即焚”条令...")
            if hasattr(self.tactical_engine, '_last_score_details_df'):
                del self.tactical_engine._last_score_details_df
            if hasattr(self.tactical_engine, '_last_risk_details_df'):
                del self.tactical_engine._last_risk_details_df
            print("        -> [焚毁完成] 临时档案已销毁，内存安全。")

    async def _run_intraday_entry_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        【V203.5 根源修正版】盘中入场确认引擎
        - 致命错误修正: 同步应用 reset_index/set_index 模式，在 merge 操作中强制保留 DatetimeIndex。
        """
        # 1. 加载配置并进行健壮性检查
        entry_params = self.tactical_engine._get_params_block('intraday_entry_params')
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

        # 3. 准备上下文信息
        setup_days_df['setup_date'] = setup_days_df.index.date
        
        trade_dates_series = pd.Series(await TradeCalendar.get_trade_dates_in_range_async(
            start_date=setup_days_df.index.min().date(),
            end_date=(setup_days_df.index.max() + pd.Timedelta(days=5)).date()
        ))
        date_map = pd.Series(trade_dates_series.iloc[1:].values, index=trade_dates_series.iloc[:-1].values)
        setup_days_df['monitoring_date'] = setup_days_df['setup_date'].map(date_map)
        setup_days_df.dropna(subset=['monitoring_date'], inplace=True)
        if setup_days_df.empty: return []

        # 4. ▼▼▼ 强制保留时间戳索引 ▼▼▼
        context_cols = ['monitoring_date', 'entry_score', 'PLATFORM_PRICE_STABLE']
        existing_context_cols = [col for col in context_cols if col in setup_days_df.columns]
        
        # 步骤 4.1: 将 minute_df 的 DatetimeIndex 显式转换为一个名为 'trade_time' 的列
        minute_df_with_ts = minute_df.reset_index().rename(columns={'index': 'trade_time'})
        minute_df_with_ts['monitoring_date'] = minute_df_with_ts['trade_time'].dt.date
        
        # 步骤 4.2: 在列上进行合并
        merged_minute_df = pd.merge(
            minute_df_with_ts,
            setup_days_df[existing_context_cols],
            on='monitoring_date',
            how='inner'
        )
        if merged_minute_df.empty: return []
        
        # 步骤 4.3: 将 'trade_time' 列恢复为 DatetimeIndex
        merged_minute_df.set_index('trade_time', inplace=True)

        # 5. 向量化计算：在合并后的分钟线数据上，一次性计算所有确认信号
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

        # 6. Groupby + idxmin: 高效找出每个监控日的“首次”确认信号
        triggered_df = merged_minute_df[final_confirmation_signal]
        if triggered_df.empty: return []
        
        # 因为索引是 DatetimeIndex, idxmin() 会返回 Timestamp 索引，可以直接用于 .loc
        first_confirmations_df = triggered_df.loc[triggered_df.groupby('monitoring_date').idxmin().iloc[:, 0]]

        # 7. 生成最终战报记录
        final_entry_records = []
        playbook_blueprints = self.tactical_engine.playbook_blueprints
        playbook_cn_map = {p['name']: p.get('cn_name', p['name']) for p in playbook_blueprints}
        
        for timestamp, row in first_confirmations_df.iterrows():
            daily_score = row.get('entry_score', 0)
            bonus_score = get_val(entry_params.get('bonus_score'), 50)
            final_score = daily_score + bonus_score
            
            playbook_name = get_val(entry_params.get('playbook_name'), 'ENTRY_INTRADAY_CONFIRMATION')
            playbook_cn_name = self.tactical_engine.scoring_params.get('metadata', {}).get(playbook_name, playbook_name)
            playbook_details = f"盘中确认: {playbook_cn_name}"

            record = self.tactical_engine._create_signal_record(
                stock_code=stock_code, 
                trade_time=timestamp, 
                timeframe=minute_tf,
                strategy_name=get_val(entry_params.get('strategy_name'), 'INTRADAY_ENTRY_CONFIRMATION'),
                signal_type='买入信号', # 明确信号类型
                final_score=final_score, # 使用正确的 final_score 字段
                risk_score=0.0,          # 盘中确认无风险
                playbook_details=playbook_details,
                close_price=row.get(f'close_{minute_tf}'),
            )
            final_entry_records.append(record)
            
        return final_entry_records

    def _run_intraday_alert_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        【V203.5 根源修正版】盘中风险预警引擎
        - 致命错误修正: 通过 reset_index/set_index 模式，在 merge 操作中强制保留 DatetimeIndex，
                        彻底解决 idxmax 返回整数并导致程序崩溃的根源性问题。
        """
        # 1. 加载配置并进行健壮性检查
        exec_params = self.tactical_engine._get_params_block('intraday_execution_params')
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
        
        upthrust_calc_params = self.tactical_engine._get_params_block('exit_strategy_params').get('upthrust_distribution_params', {})
        lookback_days = get_val(upthrust_calc_params.get('upthrust_lookback_days'), 5)
        
        is_upthrust_day = df_daily['high_D'] > df_daily['high_D'].shift(1).rolling(window=lookback_days, min_periods=1).max()
        setup_days_df = df_daily[is_upthrust_day].copy()
        if setup_days_df.empty: return []

        # 3. ▼▼▼【代码修改】修正合并逻辑，强制保留时间戳索引 ▼▼▼
        setup_days_df['monitoring_date'] = (setup_days_df.index + pd.Timedelta(days=1)).date
        
        # 步骤 3.1: 将 minute_df 的 DatetimeIndex 显式转换为一个名为 'trade_time' 的列
        minute_df_with_ts = minute_df.reset_index().rename(columns={'index': 'trade_time'})
        minute_df_with_ts['monitoring_date'] = minute_df_with_ts['trade_time'].dt.date
        
        # 步骤 3.2: 在列上进行合并
        merged_minute_df = pd.merge(minute_df_with_ts, setup_days_df[['monitoring_date']], on='monitoring_date', how='inner')
        if merged_minute_df.empty: return []
        
        # 步骤 3.3: 将 'trade_time' 列恢复为 DatetimeIndex，确保数据完整性
        merged_minute_df.set_index('trade_time', inplace=True)
        # ▲▲▲【代码修改】▲▲▲

        # 4. 向量化计算：找出所有“跌破VWAP”的时刻
        close_col, vwap_col = f'close_{minute_tf}', f'VWAP_{minute_tf}'
        if vwap_col not in merged_minute_df.columns or close_col not in merged_minute_df.columns: return []
        
        is_breaking_down = merged_minute_df[close_col] < merged_minute_df[vwap_col]
        first_breakdown_signal = is_breaking_down & ~is_breaking_down.shift(1).fillna(False)
        
        alert_days = merged_minute_df[first_breakdown_signal]['monitoring_date'].unique()
        if len(alert_days) == 0: return []

        # 5. Groupby + Apply: 对每个发生首次跌破的监控日，应用状态机逻辑
        def process_alert_day(day_df: pd.DataFrame) -> Optional[Dict]:
            is_breaking = day_df[close_col] < day_df[vwap_col]
            first_break_mask = is_breaking & ~is_breaking.shift(1).fillna(False)
            
            if not first_break_mask.any(): return None
            first_break_timestamp = first_break_mask.idxmax()
            first_alert_row = day_df.loc[first_break_timestamp]
            df_after_alert = day_df[day_df.index > first_break_timestamp]
            is_reclaimed = (df_after_alert[close_col] > df_after_alert[vwap_col]).any()
            
            if is_reclaimed:
                reclaim_time = df_after_alert[df_after_alert[close_col] > df_after_alert[vwap_col]].index[0]
                final_reason = f"[威胁解除] 曾于{first_break_timestamp.strftime('%H:%M')}跌破VWAP, 但已于{reclaim_time.strftime('%H:%M')}收复"
                final_code, final_severity = 0, 0
            else:
                final_reason = f"盘中于{first_break_timestamp.strftime('%H:%M')}跌破VWAP且至收盘未收复"
                final_code = get_val(upthrust_params.get('alert_code'), 103)
                final_severity = get_val(upthrust_params.get('severity_level'), 3)
                signal_type = '风险预警'

            return self.tactical_engine._create_signal_record(
                stock_code=stock_code, 
                trade_time=first_break_timestamp, 
                timeframe=minute_tf,
                strategy_name="INTRADAY_RISK_ALERT", 
                signal_type=signal_type, # 明确信号类型
                final_score=0.0,         # 风险预警没有进攻分
                risk_score=float(final_code), # 将风险代码作为 risk_score
                playbook_details=final_reason,
                close_price=first_alert_row[close_col],
            )

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


