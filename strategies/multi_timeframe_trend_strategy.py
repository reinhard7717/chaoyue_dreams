# 文件: strategies/multi_timeframe_trend_strategy.py
# 版本: V203.0 总指挥重构版
import re
from datetime import datetime, time
import logging
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import numpy as np

from services.indicator_services import IndicatorService
from stock_models.index import TradeCalendar
from strategies.trend_following_strategy import TrendFollowStrategy
from strategies.weekly_context_engine import WeeklyContextEngine
from utils.config_loader import load_strategy_config
from strategies.trend_following.utils import get_params_block, get_param_value
from .trend_following.intelligence_layer import MainForceState


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
        【首席法医官 V4.5：独立审计版】
        - 核心修复: 探针不再读取 DataFrame 中的 `veto_votes`，而是通过沙盘推演
                    独立计算总票数，确保报告的内部逻辑绝对一致。
        - 修复: 解决了投票详情总数与报告总数不匹配的最终BUG。
        """
        print("\n" + "="*35 + " [首席法医官 V4.5：独立审计版] " + "="*35)
        
        try:
            probe_dt = pd.to_datetime(probe_date).date()
            probe_row = df.loc[df.index.date == probe_dt].iloc[0]
            probe_ts = probe_row.name
            stock_code = probe_row.get('stock_code', 'N/A')
            if stock_code == 'N/A' and 'stock_id_D' in probe_row:
                 stock_code = probe_row['stock_id_D']
            print(f"  [案件编号]: {stock_code} @ {probe_date}")
            print(f"  [初步报告]: 进攻分={probe_row.get('entry_score', 0):.2f}, 风险分={probe_row.get('risk_score', 0):.2f}, 最终信号='{probe_row.get('signal_type', 'N/A')}'")
            print("-" * 95)
        except (IndexError, KeyError):
            print(f"  [错误] 未能在主数据流中找到目标日期 {probe_date} 的记录。调查终止。")
            print("=" * 95)
            return

        # --- 1. 沙盘推演 (Re-enacting the Vote) ---
        print("  --- 1. 联席会议投票沙盘推演 ---")
        atomic_states = self.tactical_engine.atomic_states
        probe_day_atomic = {key: series.loc[probe_ts] for key, series in atomic_states.items() if probe_ts in series.index}
        
        vote_details = []
        calculated_veto_votes = 0

        # 推演逻辑 1: 严重筹码结构风险 (3票)
        if probe_day_atomic.get('RISK_CHIP_STRUCTURE_CRITICAL_FAILURE', False):
            vote_details.append("【筹码地基审查】投出 3 票")
            calculated_veto_votes += 3

        # 推演逻辑 2: 主力行为风险 (1票)
        main_force_state_val = probe_row.get('main_force_state', -1)
        main_force_state_str = {s.value: s.name for s in MainForceState}.get(main_force_state_val, 'UNKNOWN')
        if main_force_state_str in ['DISTRIBUTING', 'COLLAPSE']:
            vote_details.append("【主力行为审查】投出 1 票")
            calculated_veto_votes += 1

        # 推演逻辑 3: 绝对否决权风险 (2票)
        veto_params = get_params_block(self.tactical_engine, 'absolute_veto_params')
        mitigation_rules = get_param_value(veto_params.get('mitigation_rules'), {})
        chip_risks_in_veto = {"RISK_CAPITAL_STRUCT_MAIN_FORCE_DISTRIBUTING", "CONTEXT_RECENT_DISTRIBUTION_PRESSURE"}
        veto_signals = [s for s in get_param_value(veto_params.get('veto_signals'), []) if s not in chip_risks_in_veto]
        for signal_name in veto_signals:
            if probe_day_atomic.get(signal_name, False):
                mitigators = mitigation_rules.get(signal_name, {}).get('mitigated_by', [])
                has_mitigator = any(probe_day_atomic.get(m, False) for m in mitigators)
                if not has_mitigator:
                    vote_details.append(f"【绝对否决权审查】投出 2 票 (原因: {signal_name})")
                    calculated_veto_votes += 2
                else:
                    print(f"    - [豁免记录] 风险 '{signal_name}' 因缓解规则被豁免，未投票。")

        # 推演逻辑 4: 常规风险 (1票)
        is_risky = probe_row.get('risk_score', 0) > probe_row.get('entry_score', 0)
        is_exempted = probe_day_atomic.get('STRUCTURE_POST_ACCUMULATION_ASCENT_C', False)
        if is_risky and not is_exempted:
            vote_details.append("【常规风险审查】投出 1 票 (原因: 风险分 > 进攻分)")
            calculated_veto_votes += 1
        
        if probe_day_atomic.get('SCORE_DYN_OPPORTUNITY_FADING', False):
            vote_details.append("【元决策审查】投出 1 票 (原因: 机会衰退)")
            calculated_veto_votes += 1
            
        if probe_day_atomic.get('SCORE_DYN_RISK_ESCALATING', False):
            vote_details.append("【元决策审查】投出 1 票 (原因: 风险抬头)")
            calculated_veto_votes += 1

        print(f"\n    [审计结果] 独立审计计算出的总否决票数为: {calculated_veto_votes}")
        if vote_details:
            print("    [投票详情]:")
            for detail in vote_details:
                print(f"      - {detail}")
        else:
            print("    - [信息] 沙盘推演未发现任何部门投出否决票。")

        # --- 2. 核心决策依据 ---
        print("\n  --- 2. 核心决策依据 ---")
        dynamic_action = probe_row.get('dynamic_action', 'N/A')
        print(f"    - [主力行为] 当日状态: {main_force_state_str} ({main_force_state_val})")
        print(f"    - [动态力学] 战术指令: {dynamic_action}")

        # --- 3. 首席法医官结论 ---
        print("\n  --- 3. 首席法医官结论 ---")
        verdict = "调查中..."
        if dynamic_action == 'AVOID':
            verdict = "【结论：真实且严重的威胁】动态力学矩阵发出了明确的'规避'指令，表明进攻动能正在衰竭而风险正在加速抬头。所有进攻信号极有可能是'牛市陷阱'或'诱多出货'。建议严格遵守规避指令。"
        elif dynamic_action == 'FORCE_ATTACK':
            verdict = "【结论：可控的良性扰动】动态力学矩阵发出了'强攻'指令，表明进攻动能正在加速而风险正在消退。当前风险大概率是主升浪中的正常洗盘或获利盘换手。进攻信号的置信度极高。"
        elif calculated_veto_votes > 0:
            reasons = [v.split('(')[0].strip() for v in vote_details]
            verdict = f"【结论：信号被否决】沙盘推演显示，信号因以下关键风险被联席会议否决（共{calculated_veto_votes}票）：{', '.join(reasons)}。基于当前规则，否决合理。"
        else:
            verdict = "【结论：高置信度买入】信号通过了所有静态和动态审查，未收到任何否决票。这是一个高置信度的进攻机会。"
            
        print(f"    {verdict}")
        print("=" * 95)

    def _run_tactical_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        【V319.0 生产模式精简版】
        - 核心优化: 移除了所有与“探针”相关的调试代码，使其成为一个纯粹的、
                    高性能的信号生成引擎，专用于生产环境。
        """
        df_daily_prepared = all_dfs.get('D_CONTEXT')
        if df_daily_prepared is None or df_daily_prepared.empty:
            print("    - [战术引擎] 日线数据为空，跳过执行。")
            return []

        try:
            # 1. 调用核心策略引擎
            daily_analysis_df, score_details_df, risk_details_df = self.tactical_engine.apply_strategy(
                df_daily_prepared, self.unified_config
            )
            
            if daily_analysis_df is None or daily_analysis_df.empty:
                print("    - [战术引擎] 引擎返回了空的分析结果。")
                self.daily_analysis_df = pd.DataFrame(index=df_daily_prepared.index)
                return []

            # 2. 【重要】保存分析结果以供其他引擎（如盘中引擎）使用
            self.daily_analysis_df = daily_analysis_df.reindex(df_daily_prepared.index)
            # 同时保存细节DataFrame，以备调试模式下使用
            self.tactical_engine._last_score_details_df = score_details_df
            self.tactical_engine._last_risk_details_df = risk_details_df
            
            # 3. 调用报告层生成数据库记录
            db_records = self.tactical_engine.prepare_db_records(
                stock_code=stock_code,
                result_df=daily_analysis_df,
                score_details_df=score_details_df,
                risk_details_df=risk_details_df,
                params=self.tactical_engine.unified_config,
                result_timeframe='D'
            )
            print(f"    -> [战术引擎] 已通过统一接口生成 {len(db_records)} 条日线信号(买入/卖出/预警)。")
            
            return db_records

        finally:
            # 注意：这里的清理逻辑保持不变，因为调试模式可能需要这些临时数据
            print("    -> [总司令部] 正在执行“阅后即焚”条令...")
            if hasattr(self.tactical_engine, '_last_score_details_df'):
                del self.tactical_engine._last_score_details_df
            if hasattr(self.tactical_engine, '_last_risk_details_df'):
                del self.tactical_engine._last_risk_details_df
            print("        -> [焚毁完成] 临时档案已销毁，内存安全。")

    async def _run_intraday_entry_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        【V203.6 修正版】盘中入场确认引擎
        - 核心修正: 修正了对 get_params_block 和 get_param_value 的调用方式。
        """
        # ▼▼▼【代码修改】: 使用导入的工具函数 ▼▼▼
        entry_params = get_params_block(self.tactical_engine, 'intraday_entry_params')
        get_val = get_param_value
        # ▲▲▲【代码修改】▲▲▲
        
        if not get_val(entry_params.get('enabled'), False): return []
        if self.daily_analysis_df is None or self.daily_analysis_df.empty: return []

        minute_tf = str(get_val(entry_params.get('timeframe'), '5'))
        minute_df = all_dfs.get(minute_tf)
        if minute_df is None or minute_df.empty: return []

        # ... (方法内后续代码保持不变) ...
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
                signal_type='买入信号',
                entry_score=final_score,
                risk_score=0.0,
                triggered_playbooks=playbook_details,
                close_price=row.get(f'close_{minute_tf}'),
                entry_signal=True,
                is_risk_warning=False
            )
            final_entry_records.append(record)
            
        return final_entry_records

    def _run_intraday_alert_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        【V203.6 修正版】盘中风险预警引擎
        - 核心修正: 修正了对 get_params_block 和 get_param_value 的调用方式。
        """
        # ▼▼▼【代码修改】: 使用导入的工具函数 ▼▼▼
        exec_params = get_params_block(self.tactical_engine, 'intraday_execution_params')
        get_val = get_param_value
        # ▲▲▲【代码修改】▲▲▲
        if not get_val(exec_params.get('enabled'), False): return []
        
        df_daily = all_dfs.get('D')
        if df_daily is None or df_daily.empty: return []

        minute_tf = str(get_val(exec_params.get('timeframe'), '30'))
        minute_df = all_dfs.get(minute_tf)
        if minute_df is None or minute_df.empty: return []

        rules_container = exec_params.get('rules', {})
        upthrust_params = rules_container.get('upthrust_rejection', {})
        if not get_val(upthrust_params.get('enabled'), False): return []
        
        # ▼▼▼【代码修改】: 使用导入的工具函数 ▼▼▼
        upthrust_calc_params = get_params_block(self.tactical_engine, 'exit_strategy_params').get('upthrust_distribution_params', {})
        # ▲▲▲【代码修改】▲▲▲
        lookback_days = get_val(upthrust_calc_params.get('upthrust_lookback_days'), 5)
        
        is_upthrust_day = df_daily['high_D'] > df_daily['high_D'].shift(1).rolling(window=lookback_days, min_periods=1).max()
        setup_days_df = df_daily[is_upthrust_day].copy()
        if setup_days_df.empty: return []

        # ... (方法内后续代码保持不变) ...
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
                reclaim_time = df_after_alert[df_after_alert[close_col] > df_after_alert[vwap_col]].index[0]
                final_reason = f"[威胁解除] 曾于{first_break_timestamp.strftime('%H:%M')}跌破VWAP, 但已于{reclaim_time.strftime('%H:%M')}收复"
                final_code, final_severity = 0, 0
            else:
                final_reason = f"盘中于{first_break_timestamp.strftime('%H:%M')}跌破VWAP且至收盘未收复"
                final_code = get_val(upthrust_params.get('alert_code'), 103)
                final_severity = get_val(upthrust_params.get('severity_level'), 3)

            return self.tactical_engine._create_signal_record(
                stock_code=stock_code, 
                trade_time=first_break_timestamp, 
                timeframe=minute_tf,
                strategy_name="INTRADAY_RISK_ALERT", 
                signal_type=signal_type,
                entry_score=float(final_code),
                risk_score=float(final_code),
                triggered_playbooks=final_reason,
                close_price=first_alert_row[close_col],
                entry_signal=False,
                is_risk_warning=True
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
        【V319.0 探针专属版】
        - 核心重构: 将“探针”的部署逻辑完全移入此方法，与生产流程解耦。
        - 新流程:
          1. 正常执行 `run_for_stock` 以生成所有分析结果。
          2. 从 `tactical_engine` 中获取最后一次运行的详细分析数据。
          3. 检查配置文件，如果设置了 `probe_date`，则启动探针进行深度解剖。
          4. 最后，展示全流程的信号透视报告。
        """
        print("=" * 80)
        print(f"--- [历史回溯调试启动 (V319.0 探针专属版)] ---")
        print(f"    -> 股票代码: {stock_code}")
        print(f"    -> 回测时段: {start_date} to {end_date}")
        print("=" * 80)

        try:
            # 步骤 1: 正常执行核心流程，生成所有数据
            all_records = await self.run_for_stock(stock_code, trade_time=end_date)
            if all_records is None: return

            # 步骤 2: 检查是否需要部署探针
            debug_params = get_params_block(self.tactical_engine, 'debug_params')
            probe_date = get_param_value(debug_params.get('probe_date'))
            
            if probe_date:
                print(f"\n    --- [总司令部] 接到密令！正在对 {probe_date} 的战况进行深度解剖... ---")
                # 从战术引擎获取最后一次运行的详细结果
                last_df = self.daily_analysis_df
                last_score_details = getattr(self.tactical_engine, '_last_score_details_df', pd.DataFrame())
                last_risk_details = getattr(self.tactical_engine, '_last_risk_details_df', pd.DataFrame())

                if last_df is not None and not last_df.empty:
                    self._deploy_field_coroner_probe(
                        df=last_df,
                        probe_date=probe_date,
                        score_details=last_score_details,
                        risk_details=last_risk_details
                    )
                else:
                    print("    -> [探针错误] 未能获取到有效的分析数据帧，无法部署探针。")

            # 步骤 3: 展示全流程信号透视报告
            print(f"\n[步骤 2/2] 正在筛选并展示目标时段 ({start_date} to {end_date}) 的所有信号...")
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
                # ... (此处的日志打印逻辑保持不变) ...
                time_obj = pd.to_datetime(record['trade_time'])
                time_str = time_obj.strftime('%Y-%m-%d %H:%M:%S %Z')
                tf = record.get('timeframe', 'N/A')
                signal_type = "未知信号"
                details = "无详细信息"
                
                reason = record.get('exit_signal_reason') or record.get('triggered_playbooks') or "原因未知"

                if record.get('exit_signal_code', 0) > 0:
                    severity = record.get('exit_severity_level', 0)
                    signal_type = f"卖出警报(L{severity})"
                    details = f"风险分: {record.get('risk_score', 0):<3.0f} | 原因: {reason}"
                
                elif record.get('signal_type') == '卖出信号':
                    risk_score = record.get('risk_score', 0.0)
                    playbooks_raw = record.get('triggered_playbooks', '无剧本信息')
                    playbooks_str = re.sub(r'\s+', ' ', playbooks_raw).strip()
                    details = f"风险分: {risk_score:<7.2f} | 剧本: {playbooks_str}"
                    signal_type = "综合卖出"

                elif record.get('entry_signal'):
                    score = record.get('entry_score', 0.0)
                    playbooks_raw = record.get('triggered_playbooks', '无剧本信息')
                    playbooks_str = re.sub(r'\s+', ' ', playbooks_raw).strip()
                    details = f"得分: {score:<7.2f} | 剧本: {playbooks_str}"
                    signal_type = "综合买入"
                
                elif record.get('is_risk_warning'):
                    signal_type = "风险预警"
                    details = f"风险分: {record.get('risk_score', 0):<3.0f} | 原因: {reason}"
                
                elif record.get('strategy_name') == 'INTRADAY_RISK_ALERT':
                    signal_type = f"盘中异动"
                    details = f"原因: {reason}"

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


