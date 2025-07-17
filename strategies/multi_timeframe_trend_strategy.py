# 文件: strategies/multi_timeframe_trend_strategy.py
import io       # 导入 io
import sys      # 导入 sys
import re       # 导入 re
from contextlib import redirect_stdout # 导入 redirect_stdout
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
from strategies.weekly_trend_follow_strategy import WeeklyTrendFollowStrategy
from utils.config_loader import load_strategy_config
from utils.data_sanitizer import sanitize_for_json

logger = logging.getLogger(__name__)

class MultiTimeframeTrendStrategy:

    def __init__(self):
        """
        【V8.0 统一指挥改造版】
        - 核心升级: 废除所有配置合并逻辑，改为加载并分发单一的统一配置文件。
        - 流程重构: 引擎初始化流程被极大简化，直接从统一配置中提取所需部分。
        """
        unified_config_path = 'config/trend_follow_strategy.json'
        self.unified_config = load_strategy_config(unified_config_path)
        self.indicator_service = IndicatorService()
        # 1. 初始化战略参谋部 (周线上下文引擎)
        #    从统一配置中，精准提取 'weekly_context_params' 部分传给它
        self.strategic_engine = WeeklyContextEngine(
            config=self.unified_config
        )
        # 2. 初始化一线作战部队 (日线战术引擎)
        #    将整个统一配置传给它，由它自行按需取用
        self.tactical_engine = TrendFollowStrategy(
            config=self.unified_config
        )        
        self.daily_analysis_df = None
        # 确保时间周期发现逻辑使用新的统一配置对象
        self.required_timeframes = self.indicator_service._discover_required_timeframes_from_config(self.unified_config)

    # ▼▼▼ 标准化战报生成器 ▼▼▼
    def _create_signal_record(self, **kwargs) -> Dict[str, Any]:
        """
        【V137.0 标准化战报生成器】
        - 核心升级: 将 stable_platform_price 正式加入标准战报模板。
        - 收益: 建立了一个全系统统一的数据契约，确保任何信号记录都包含所有关键字段。
        """
        trade_time_input = kwargs.get('trade_time')
        if trade_time_input is None:
            raise ValueError("创建信号记录时必须提供 'trade_time'")
        
        ts = pd.to_datetime(trade_time_input, utc=True)
        if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
            standard_trade_time = ts.tz_localize('Asia/Shanghai').tz_convert('UTC').to_pydatetime()
        else:
            standard_trade_time = ts.tz_convert('UTC').to_pydatetime()

        # 这就是“标准化战报条例”的模板
        record = {
            "stock_code": None,
            "trade_time": standard_trade_time,
            "timeframe": "N/A",
            "strategy_name": "UNKNOWN",
            "close_price": 0.0,
            "entry_score": 0.0,
            "entry_signal": False,
            "exit_signal_code": 0,
            "exit_severity_level": 0,
            "exit_signal_reason": None,
            "triggered_playbooks": [],
            "triggered_playbooks_cn": [], # 新增，用于存储中文剧本
            "stable_platform_price": None, # 【核心新增】将平台价格纳入标准
            "context_snapshot": {},
            "analysis_text": None
        }
        
        record.update(kwargs)
        
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

    async def run_for_stock(self, stock_code: str, trade_time: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        【V8.0 统一指挥版】
        - 核心流程: 数据准备 -> 周线分析 -> 信号注入 -> 日线决策 -> 汇总报告
        - 适配修改: 使用 self.unified_config 作为数据准备的唯一配置源。
        """
        print(f"--- 开始为【{stock_code}】执行联合作战分析 (V8.0) ---")
        
        # 步骤1: 调用数据服务，获取所有锻造好的数据
        print("  -> [步骤1/4] 正在调用 IndicatorService 获取所有周期数据...")
        all_dfs = await self.indicator_service.prepare_data_for_strategy(
            stock_code, self.unified_config, trade_time
        )

        if 'D' not in all_dfs or 'W' not in all_dfs:
            print(f"    - 错误: 缺少日线或周线数据，无法执行联合作战。")
            return None
            
        # 步骤2: 调用战略参谋部，生成周线战略信号
        print("  -> [步骤2/4] 正在调用 WeeklyContextEngine 生成周线战略信号...")
        df_weekly_context = self.strategic_engine.generate_context(all_dfs['W'])
        
        # 步骤3: 将周线战略信号注入日线数据
        print("  -> [步骤3/4] 正在将周线战略背景注入日线数据...")
        df_daily_enhanced = self._merge_strategic_context_to_daily(all_dfs['D'], df_weekly_context)
        
        all_dfs['D'] = df_daily_enhanced
        
        # 步骤4: 在增强后的数据上，运行所有战术及风险引擎
        print("  -> [步骤4/4] 正在运行日线战术引擎及其他风险引擎...")
        tactical_records = self._run_tactical_engine(stock_code, all_dfs)
        risk_alert_records = self._run_intraday_alert_engine(stock_code, all_dfs)
        intraday_entry_records = await self._run_intraday_entry_engine(stock_code, all_dfs)
        
        all_records = tactical_records + risk_alert_records + intraday_entry_records
        
        print(f"--- 所有引擎分析完毕，共生成 {len(all_records)} 条最终信号记录准备交付。 ---")
        return all_records

    def _merge_strategic_context_to_daily(self, df_daily: pd.DataFrame, df_weekly_context: pd.DataFrame) -> pd.DataFrame:
        """
        【V7.0 技术升级版】
        - 核心升级: 放弃 merge_asof，改用更精准、更健壮的 reindex + ffill 技术。
        - 收益: 完美地将周线信号广播到该周的每一个交易日，逻辑更清晰，结果更可靠。
        """
        if df_weekly_context is None or df_weekly_context.empty:
            print("    - [情报融合] 周线引擎未返回任何战略信号，跳过注入。")
            return df_daily
        
        print(f"    - [情报融合] 准备将 {len(df_weekly_context.columns)} 个周线信号注入日线数据...")
        
        # 步骤1: 使用 reindex 将周线信号的索引扩展到日线级别
        # method='ffill' (forward-fill) 会将周一的信号值填充到周二、三、四、五
        df_weekly_aligned = df_weekly_context.reindex(df_daily.index, method='ffill')
        
        # 步骤2: 使用 merge 合并，比 join 更安全，可以处理列名冲突
        df_merged = df_daily.merge(df_weekly_aligned, left_index=True, right_index=True, how='left', suffixes=('', '_weekly_dup'))
        
        # 步骤3: 指令翻译与分发 (逻辑与旧版类似，但更简洁)
        # 遍历所有从周线合并过来的列
        for col in df_weekly_context.columns:
            if col not in df_merged.columns: continue
            
            # 简单的类型填充
            if col.startswith(('playbook_', 'signal_', 'state_', 'event_', 'filter_')):
                df_merged[col] = df_merged[col].fillna(False).astype(bool)
            elif col.startswith(('washout_score_', 'rejection_signal_')):
                df_merged[col] = df_merged[col].fillna(0).astype(int)
        
        print("    - [情报融合] 注入完成。日线数据已获得周线战略指令加持。")
        return df_merged

    # 情报融合中心
    def _fuse_intraday_confirmations(self, daily_records: List[Dict], confirmation_records: List[Dict]) -> List[Dict]:
        """
        【V117.11 新增】情报融合中心。
        - 核心职责: 将分钟线的“确认信号”融合进原始的“日线信号”记录中，而不是创建新纪录。
        - 工作流程:
          1. 以日线信号的日期为键，创建一个快速查找字典。
          2. 遍历所有分钟线确认信号。
          3. 如果找到对应日期的日线信号，则用分钟线确认信号的更高分数、新剧本等信息，
             直接“覆盖升级”原始的日线记录。
        - 收益: 确保了最终输出的信号是唯一的、经过分钟线增强的最终决策，解决了分数不更新的问题。
        """
        if not confirmation_records:
            return daily_records

        # 创建一个以日期为键的日线记录查找字典，提高效率
        daily_lookup = {
            pd.to_datetime(rec['trade_time'], utc=True).date(): rec
            for rec in daily_records if rec.get('entry_signal')
        }
        
        fused_dates = set()

        for conf_rec in confirmation_records:
            conf_date = pd.to_datetime(conf_rec['trade_time'], utc=True).date()
            
            # 如果在日线记录中找到了对应的日期
            if conf_date in daily_lookup:
                daily_rec_to_update = daily_lookup[conf_date]
                
                # 使用分钟确认信号的关键信息来“升级”日线信号
                daily_rec_to_update['entry_score'] = conf_rec['entry_score']
                daily_rec_to_update['triggered_playbooks'] = conf_rec['triggered_playbooks']
                daily_rec_to_update['context_snapshot'] = conf_rec['context_snapshot']
                # 可以在这里添加一个标记，表示此信号已被分钟线确认
                daily_rec_to_update['context_snapshot']['intraday_confirmed'] = True
                
                fused_dates.add(conf_date)
                logger.info(f"    - [情报融合] 日期 {conf_date} 的日线信号已被分钟线确认，分数已升级至 {conf_rec['entry_score']:.0f}。")

        # 对于那些没有被分钟线确认的日线信号，我们仍然保留它们
        return list(daily_lookup.values())

    async def _run_intraday_entry_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        【V148.0 情报链路完整性修复版】
        - 核心修复: 解决了因本函数在重新生成信号记录时，未进行剧本名称的
                    中英文翻译，导致最终报告无法显示剧本名称的BUG。
        - 新逻辑:
          1. 在函数内部，主动获取战术引擎的剧本蓝图。
          2. 创建一个中英文名称的映射字典 (playbook_cn_name_map)。
          3. 在提取出英文剧本列表后，立刻使用映射字典生成中文剧本列表。
          4. 将中、英文两个列表同时传递给 _create_signal_record 生成器。
        - 收益: 确保了所有经此模块处理的“精加工”信号，都包含完整的中英文剧本信息。
        """
        final_entry_records = []
        entry_params = self.tactical_engine._get_params_block(self.unified_config, 'intraday_entry_params', {})
        get_val = self.tactical_engine._get_param_value
        
        if not get_val(entry_params.get('enabled'), False): return []
        if self.daily_analysis_df is None or self.daily_analysis_df.empty: return []

        # 【新增】获取剧本蓝图并创建名称映射字典
        playbook_blueprints = self.tactical_engine.playbook_blueprints
        playbook_cn_name_map = {p['name']: p.get('cn_name', p['name']) for p in playbook_blueprints}

        daily_score_threshold = get_val(entry_params.get('daily_score_threshold'), 100)
        setup_days_df = self.daily_analysis_df[self.daily_analysis_df['entry_score'] >= daily_score_threshold]
        
        if setup_days_df.empty: return []
            
        minute_tf = str(get_val(entry_params.get('timeframe'), '5'))
        minute_df = all_dfs.get(minute_tf)
        
        rules = entry_params.get('confirmation_rules', {})
        min_time_after_open = get_val(rules.get('min_time_after_open'), 15)

        for setup_date_ts, setup_row in setup_days_df.iterrows():
            setup_date = setup_date_ts.date()
            daily_score = setup_row.get('entry_score', 0)
            
            daily_playbooks = []
            if self.tactical_engine._last_score_details_df is not None and setup_date_ts in self.tactical_engine._last_score_details_df.index:
                score_details_for_day = self.tactical_engine._last_score_details_df.loc[setup_date_ts]
                playbooks_with_scores = score_details_for_day[score_details_for_day > 0]
                # 过滤掉非剧本的得分项
                excluded_prefixes = ('BASE_', 'BONUS_', 'PENALTY_', 'INDUSTRY_', 'WATCHING_SETUP', 'CHIP_PURITY_MULTIPLIER', 'VOLATILITY_SILENCE_MULTIPLIER')
                daily_playbooks = [ item for item in playbooks_with_scores.index if not item.startswith(excluded_prefixes) ]

            # 【新增】生成中文剧本列表
            daily_playbooks_cn = [playbook_cn_name_map.get(item, item) for item in daily_playbooks]

            platform_price_on_setup_day = setup_row.get('PLATFORM_PRICE_STABLE', None)
            decision_time = datetime.combine(setup_date, time(16, 0))

            default_daily_record = self._create_signal_record(
                stock_code=stock_code,
                trade_time=decision_time,
                timeframe='D',
                strategy_name=daily_playbooks[0] if daily_playbooks else "DAILY_ENTRY_SIGNAL",
                close_price=setup_row.get('close_D', 0),
                entry_score=daily_score,
                entry_signal=True,
                triggered_playbooks=daily_playbooks,
                triggered_playbooks_cn=daily_playbooks_cn, # 【新增】传递中文剧本列表
                stable_platform_price=platform_price_on_setup_day,
                context_snapshot={'close': setup_row.get('close_D', 0), 'daily_score': daily_score, 'intraday_confirmed': False},
            )

            final_record_for_this_setup = default_daily_record
            minute_confirmation_found = False

            if minute_df is not None and not minute_df.empty:
                monitoring_date = await TradeCalendar.get_next_trade_date_async(reference_date=setup_date)
                
                if monitoring_date:
                    alert_day_minute_df = minute_df[minute_df.index.date == monitoring_date].copy()
                    if not alert_day_minute_df.empty:
                        # ... 分钟线确认逻辑保持不变 ...
                        final_confirmation_signal = pd.Series(True, index=alert_day_minute_df.index)
                        close_col_m = f'close_{minute_tf}'
                        vwap_rule = rules.get('vwap_reclaim', {})
                        if get_val(vwap_rule.get('enabled'), False):
                            vwap_col_m = f'VWAP_{minute_tf}'
                            if vwap_col_m in alert_day_minute_df.columns and close_col_m in alert_day_minute_df.columns:
                                final_confirmation_signal &= (alert_day_minute_df[close_col_m] > alert_day_minute_df[vwap_col_m])

                        vol_rule = rules.get('volume_confirmation', {})
                        if get_val(vol_rule.get('enabled'), False):
                            volume_col_m = f'volume_{minute_tf}'
                            vol_ma_period = get_val(vol_rule.get('ma_period'), 21)
                            vol_ma_col_m = f'VOL_MA_{vol_ma_period}_{minute_tf}'
                            if vol_ma_col_m in alert_day_minute_df.columns and volume_col_m in alert_day_minute_df.columns:
                                final_confirmation_signal &= (alert_day_minute_df[volume_col_m] > alert_day_minute_df[vol_ma_col_m])
                        
                        naive_market_open_time = datetime.combine(monitoring_date, time(9, 30))
                        aware_market_open_time = pd.Timestamp(naive_market_open_time, tz='Asia/Shanghai')
                        monitoring_start_time = aware_market_open_time + pd.Timedelta(minutes=min_time_after_open)
                        
                        triggered_minutes = alert_day_minute_df[(alert_day_minute_df.index >= monitoring_start_time) & (final_confirmation_signal == True)]
                        
                        if not triggered_minutes.empty:
                            minute_confirmation_found = True
                            first_confirmation_minute = triggered_minutes.iloc[0]
                            confirm_time = first_confirmation_minute.name
                            confirm_price = first_confirmation_minute[close_col_m]
                            bonus_score = get_val(entry_params.get('bonus_score'), 50)
                            final_score = daily_score + bonus_score
                            
                            # 【新增】为分钟线确认信号也生成中英文剧本列表
                            confirmed_playbooks = list(set(daily_playbooks + [get_val(entry_params.get('playbook_name'), 'ENTRY_INTRADAY_CONFIRMATION')]))
                            confirmed_playbooks_cn = [playbook_cn_name_map.get(item, item) for item in confirmed_playbooks]

                            confirmation_record = self._create_signal_record(
                                stock_code=stock_code,
                                trade_time=confirm_time,
                                timeframe=minute_tf,
                                strategy_name=get_val(entry_params.get('strategy_name'), 'INTRADAY_ENTRY_CONFIRMATION'),
                                close_price=confirm_price,
                                entry_score=final_score,
                                entry_signal=True,
                                triggered_playbooks=confirmed_playbooks,
                                triggered_playbooks_cn=confirmed_playbooks_cn, # 【新增】传递中文剧本列表
                                stable_platform_price=platform_price_on_setup_day,
                                context_snapshot={'close': confirm_price, 'daily_score': daily_score, 'bonus': bonus_score, 'intraday_confirmed': True, 'setup_date': setup_date.strftime('%Y-%m-%d')},
                            )
                            final_record_for_this_setup = confirmation_record

            final_entry_records.append(final_record_for_this_setup)

        return final_entry_records

    def _run_intraday_alert_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        【V128.0 动态战情更新版】
        - 核心升级: 彻底重构VWAP警报逻辑，引入“状态机”概念。
        - 新逻辑:
          1. 找出所有“跌破VWAP”的初步警报时刻。
          2. 对于每个初步警报，向后追踪当天的分钟线数据。
          3. 如果在当天收盘前，价格重新站回VWAP之上，则更新原始警报记录，
             将其标记为“威胁已解除”，并将风险码置为0。
          4. 如果直到收盘都未能收复，则保留原始警报。
        - 收益: 解决了警报发出后无法根据后续行情动态更新或撤销的问题，
                使得最终的风险警报能更准确地反映收盘时的真实状态。
        """
        all_alerts = []
        exec_params = self.tactical_engine._get_params_block(self.unified_config, 'intraday_execution_params', {})
        get_val = self.tactical_engine._get_param_value
        
        if not get_val(exec_params.get('enabled'), False):
            logger.info("    - [风险预警任务] 跳过，因为 'intraday_execution_params' 在配置中被禁用。")
            return []
        
        df_daily = all_dfs.get('D')
        if df_daily is None or df_daily.empty:
            logger.warning(f"[{stock_code}] 缺少日线数据，分钟线风险预警引擎跳过。")
            return []

        rules_container = exec_params.get('rules', {})
        upthrust_rejection_params = rules_container.get('upthrust_rejection', {})
        
        if not get_val(upthrust_rejection_params.get('enabled'), False):
            logger.info("    - [风险预警任务] 跳过 'upthrust_rejection' 规则，因为它在配置中被禁用。")
            return []

        upthrust_calc_params = self.tactical_engine._get_params_block(self.unified_config, 'exit_strategy_params', {}).get('upthrust_distribution_params', {})
        lookback_days = get_val(upthrust_calc_params.get('upthrust_lookback_days'), 5)
        
        is_upthrust_day = df_daily['high_D'] > df_daily['high_D'].shift(1).rolling(window=lookback_days, min_periods=1).max()
        
        setup_days_df = df_daily[is_upthrust_day]
        if setup_days_df.empty:
            return []
            
        logger.info(f"    - [风险预警任务] 发现 {len(setup_days_df)} 个风险预备日，启动次日盘中监控...")

        minute_tf = str(get_val(exec_params.get('timeframe'), '30'))
        minute_df = all_dfs.get(minute_tf)
        if minute_df is None or minute_df.empty:
            logger.warning(f"[{stock_code}] 缺少 {minute_tf} 分钟线数据，分钟线风险预警引擎跳过。")
            return []

        for setup_date, setup_row in setup_days_df.iterrows():
            monitoring_date = (setup_date + pd.Timedelta(days=1)).date()
            alert_day_minute_df = minute_df[minute_df.index.date == monitoring_date].copy()
            if alert_day_minute_df.empty:
                continue

            close_col = f'close_{minute_tf}'
            vwap_col = f'VWAP_{minute_tf}'
            if vwap_col not in alert_day_minute_df.columns or close_col not in alert_day_minute_df.columns:
                continue

            # 找出所有“首次跌破”的时刻
            is_breaking_down = (alert_day_minute_df[close_col] < alert_day_minute_df[vwap_col])
            first_breakdown_signal = is_breaking_down & ~is_breaking_down.shift(1).fillna(False)
            
            triggered_minutes = alert_day_minute_df[first_breakdown_signal]

            if not triggered_minutes.empty:
                first_alert_minute_row = triggered_minutes.iloc[0]
                alert_time = first_alert_minute_row.name
                alert_price = first_alert_minute_row[close_col]
                
                playbook_name = get_val(upthrust_rejection_params.get('alert_playbook_name'), 'EXIT_INTRADAY_UPTHRUST_REJECTION')
                alert_code = get_val(upthrust_rejection_params.get('alert_code'), 103)
                severity_level = get_val(upthrust_rejection_params.get('severity_level'), 3)

                # --- 步骤2: 战情动态跟踪 ---
                # 检查在警报发出后，当天是否重新站回VWAP
                df_after_alert = alert_day_minute_df[alert_day_minute_df.index > alert_time]
                is_reclaimed = (df_after_alert[close_col] > df_after_alert[vwap_col])
                reclaim_minutes = df_after_alert[is_reclaimed]

                final_reason = f"盘中跌破VWAP, 由前一日({setup_date.date()})创近期新高触发监控"
                final_alert_code = alert_code
                final_severity = severity_level

                if not reclaim_minutes.empty:
                    # 如果找到了收复的时刻
                    first_reclaim_row = reclaim_minutes.iloc[0]
                    reclaim_time = first_reclaim_row.name
                    
                    # --- 步骤3: 更新警报状态为“威胁已解除” ---
                    final_reason = f"[威胁解除] 曾于{alert_time.strftime('%H:%M')}跌破VWAP, 但已于{reclaim_time.strftime('%H:%M')}收复"
                    final_alert_code = 0  # 将风险码置为0，表示无效警报
                    final_severity = 0    # 严重等级也置为0
                    # print(f"         - [战情更新] 日期: {monitoring_date} | 威胁已于 {reclaim_time.time()} 解除。")
                else:
                    # 如果直到收盘都未收复，保留原始警报
                    # print(f"         - [风险警报!] 日期: {monitoring_date} | 时间: {alert_time.time()} | 价格: {alert_price:.2f} | 规则: {playbook_name} (威胁未解除)")
                    pass

                # 无论是否解除，都创建一条记录，以便追踪战况
                record = self._create_signal_record(
                    stock_code=stock_code,
                    trade_time=alert_time, # 记录原始警报时间
                    timeframe=minute_tf,
                    strategy_name="INTRADAY_RISK_ALERT",
                    close_price=alert_price,
                    entry_signal=False,
                    exit_signal_code=final_alert_code,
                    exit_severity_level=final_severity,
                    exit_signal_reason=final_reason,
                    triggered_playbooks=[playbook_name],
                    context_snapshot={'close': alert_price, 'vwap': first_alert_minute_row[vwap_col], 'reclaimed': not reclaim_minutes.empty},
                )
                all_alerts.append(record)

        return all_alerts

    def _merge_and_deduplicate_signals(self, daily_records: List[Dict], intraday_records: List[Dict]) -> List[Dict]:
        if not daily_records and not intraday_records:
            return daily_records or intraday_records
        signals_by_day = defaultdict(dict)
        def get_trade_date(trade_time_value: Any) -> Optional[datetime.date]:
            try:
                if isinstance(trade_time_value, str):
                    return pd.to_datetime(trade_time_value, utc=True).date()
                elif hasattr(trade_time_value, 'date'):
                    return trade_time_value.date()
                else:
                    return None
            except Exception as e:
                return None
        for record in daily_records:
            if record.get('entry_signal'):
                trade_date = get_trade_date(record.get('trade_time'))
                if trade_date:
                    signals_by_day[trade_date]['D'] = record
        for record in intraday_records:
            if record.get('entry_signal'):
                trade_date = get_trade_date(record.get('trade_time'))
                if trade_date:
                    signals_by_day[trade_date]['M'] = record
        final_records = []
        sorted_dates = sorted(signals_by_day.keys())
        for trade_date in sorted_dates:
            signals = signals_by_day[trade_date]
            if 'M' in signals:
                final_records.append(signals['M'])
            elif 'D' in signals:
                final_records.append(signals['D'])
        return final_records

    def _run_strategic_engine(self, df_weekly: pd.DataFrame) -> pd.DataFrame:
        if df_weekly is None or df_weekly.empty:
            logger.warning("周线数据为空，战略引擎跳过。")
            return pd.DataFrame()
        return self.strategic_engine.apply_strategy(df_weekly)

    def _merge_strategic_signals_to_daily(self, df_daily: pd.DataFrame, strategic_signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        【V6.13 净化合并版】
        - 核心修复: 从根源上解决日线DataFrame被周线数据污染的问题。
          在合并前，对周线信号DataFrame进行“净化”，只保留明确需要传递给日线引擎的、
          以 '_W' 结尾或特定前缀开头的信号列，防止任何可能冲突的列被合并。
        """
        if strategic_signals_df is None or strategic_signals_df.empty:
            return df_daily
        
        print("---【总指挥-指令分发 V6.13】开始将周线战略信号翻译并注入日线数据... ---")
        df_daily_copy = df_daily.copy()

        # --- 步骤1: 净化周线信号，只保留需要的列 ---
        # 定义需要保留的周线信号列的规则
        cols_to_keep = [
            col for col in strategic_signals_df.columns 
            if col.endswith('_W') or col.startswith(('playbook_', 'signal_', 'state_', 'event_', 'filter_', 'washout_score_', 'rejection_signal_'))
        ]
        
        # 如果没有找到任何需要保留的列，则直接返回原始日线df
        if not cols_to_keep:
            print("    - [警告] 在周线信号中未找到任何需要合并的列，跳过合并。")
            return df_daily_copy

        # 创建一个只包含所需信号的干净的周线DataFrame
        clean_strategic_df = strategic_signals_df[cols_to_keep].copy()
        print(f"    - [净化] 从周线DataFrame中筛选出 {len(cols_to_keep)} 个信号列进行合并。")
        
        # --- 步骤2: 使用净化后的周线信号进行合并 ---
        df_merged = pd.merge_asof(
            left=df_daily_copy.sort_index(), 
            right=clean_strategic_df.sort_index(), # 使用净化后的DataFrame
            left_index=True, 
            right_index=True, 
            direction='backward'
        )
        
        # --- 步骤3: 指令翻译与分发 (逻辑不变) ---
        # 遍历所有从周线合并过来的列
        for col in clean_strategic_df.columns: # 遍历净化后的列
            if col not in df_merged.columns: continue
            
            if col == 'signal_breakout_trigger_W':
                new_col_name = 'BASE_SIGNAL_BREAKOUT_TRIGGER'
                df_merged.rename(columns={col: new_col_name}, inplace=True)
                df_merged[new_col_name] = df_merged[new_col_name].fillna(False).astype(bool)
                print(f"    - [指令分发] 原始信号 '{col}' 已翻译为 -> '{new_col_name}'")
            elif col == 'playbook_coppock_stabilizing_W':
                new_col_name = 'CONTEXT_STRATEGIC_BOTTOMING_W'
                df_merged.rename(columns={col: new_col_name}, inplace=True)
                df_merged[new_col_name] = df_merged[new_col_name].fillna(False).astype(bool)
                print(f"    - [指令分发] 原始信号 '{col}' 已翻译为 -> '{new_col_name}' (左侧观察许可)")
            elif col == 'playbook_coppock_accelerating_W':
                new_col_name = 'EVENT_STRATEGIC_ACCELERATING_W'
                df_merged.rename(columns={col: new_col_name}, inplace=True)
                df_merged[new_col_name] = df_merged[new_col_name].fillna(False).astype(bool)
                print(f"    - [指令分发] 原始信号 '{col}' 已翻译为 -> '{new_col_name}' (右侧加速事件)")
            elif col.startswith(('playbook_', 'signal_', 'state_', 'event_', 'filter_')):
                df_merged[col] = df_merged[col].fillna(False).astype(bool)
            elif col.startswith(('washout_score_', 'rejection_signal_')):
                df_merged[col] = df_merged[col].fillna(0).astype(int)
                
        print("---【总指挥-指令分发】完成。日线数据已获得周线战略指令加持。 ---")
        return df_merged

    def _run_tactical_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        【V202.0 终极战报版】
        - 核心升级: 新增对“每日风险书记官”(_generate_daily_risk_signals)的调用，
                    并将所有类型的信号（入场、风险、持仓管理）整合到一份完整的战报中。
        """
        df_daily_prepared = all_dfs.get('D')
        if df_daily_prepared is None or df_daily_prepared.empty:
            return []

        print(f"--- [引擎2-调试] 战术引擎接收到的日线数据时间范围到: {df_daily_prepared.index.max()}")

        daily_analysis_df, atomic_signals = self.tactical_engine.apply_strategy(df_daily_prepared, self.unified_config)
        
        if daily_analysis_df is None or daily_analysis_df.empty:
            print("--- [引擎2-调试] 战术引擎返回了空的分析结果。")
            self.daily_analysis_df = pd.DataFrame(index=df_daily_prepared.index)
            return []
        
        print(f"--- [引擎2-调试] 战术引擎原始分析结果时间范围到: {daily_analysis_df.index.max()}")

        self.daily_analysis_df = daily_analysis_df.reindex(df_daily_prepared.index)
        if 'entry_score' in self.daily_analysis_df.columns:
            self.daily_analysis_df['entry_score'].fillna(0, inplace=True)
        bool_cols = self.daily_analysis_df.select_dtypes(include='bool').columns
        for col in bool_cols:
            self.daily_analysis_df[col].fillna(False, inplace=True)
        
        print(f"--- [引擎2-调试] reindex后，最终 self.daily_analysis_df 时间范围到: {self.daily_analysis_df.index.max()}")

        # 1. 获取“主要事件”记录（买入信号）
        entry_records = self.tactical_engine.prepare_db_records(
            stock_code, self.daily_analysis_df, atomic_signals, 
            params=self.unified_config, result_timeframe='D'
        )
        print(f"    -> [战报记录] 已生成 {len(entry_records)} 条主要事件(买入)记录。")

        # ▼▼▼【代码修改 V202.0】: 新增对每日风险信号的生成和整合 ▼▼▼
        # 2. 调用“每日风险书记官”，生成风险和出场信号记录
        risk_and_exit_records = self._generate_daily_risk_signals(
            stock_code, self.daily_analysis_df, self.unified_config
        )
        print(f"    -> [战报记录] 已生成 {len(risk_and_exit_records)} 条风险/出场信号记录。")

        # 3. 提取持仓管理引擎生成的“过程记录”
        position_management_records = []
        pm_cols = ['trade_action', 'alert_level', 'alert_reason', 'position_size']
        if all(col in self.daily_analysis_df.columns for col in pm_cols):
            process_df = self.daily_analysis_df[
                (self.daily_analysis_df['trade_action'] != '') &
                (~self.daily_analysis_df['trade_action'].isin(['ENTRY', 'HOLD'])) &
                (~self.daily_analysis_df['trade_action'].str.startswith('EXIT'))
            ].copy()
            for timestamp, row in process_df.iterrows():
                record = self._create_signal_record(
                    stock_code=stock_code, trade_time=timestamp, timeframe='D',
                    strategy_name="POSITION_MANAGEMENT", close_price=row.get('close_D'),
                    trade_action=row.get('trade_action'), alert_level=row.get('alert_level'),
                    alert_reason=row.get('alert_reason'), position_size=row.get('position_size'),
                )
                position_management_records.append(record)
            if position_management_records:
                 print(f"    -> [战报记录] 已提取 {len(position_management_records)} 条持仓过程记录(如减仓/预警)。")

        # 4. 合并所有记录
        all_records = entry_records + risk_and_exit_records + position_management_records
        # ▲▲▲【代码修改 V202.0】▲▲▲

        # 情报下放逻辑保持不变
        cols_to_broadcast = ['PLATFORM_PRICE_STABLE'] 
        existing_cols_to_broadcast = [col for col in cols_to_broadcast if col in self.daily_analysis_df.columns]
        if existing_cols_to_broadcast:
            print(f"    -> [情报下放] 准备将日线情报 {existing_cols_to_broadcast} 下发至分钟线...")
            broadcast_df = self.daily_analysis_df[existing_cols_to_broadcast].copy()
            for tf, df_intraday in all_dfs.items():
                if tf.isdigit():
                    all_dfs[tf] = pd.merge_asof(
                        left=df_intraday.sort_index(), right=broadcast_df.sort_index(),
                        left_index=True, right_index=True, direction='backward'
                    )
                    print(f"      - 情报已成功注入 {tf} 分钟数据。")
        
        return all_records

    # ▼▼▼ “每日风险书记官”方法 ▼▼▼
    def _generate_daily_risk_signals(self, stock_code: str, daily_analysis_df: pd.DataFrame, params: dict) -> List[Dict[str, Any]]:
        """
        【V202.1 触发逻辑修复版】每日风险书记官
        - 核心修复: 筛选逻辑不再依赖于 `exit_signal_code`，而是直接使用 `risk_score > 0`
                    作为判断标准。这确保了只要当天存在任何风险分，就会被捕获并生成
                    一条对应的风险信号记录，解决了有风险分但无信号输出的问题。
        """
        # 1. 获取风险剧本的定义，用于翻译原因
        risk_playbooks = self.tactical_engine._get_risk_playbook_blueprints()
        risk_playbook_map = {bp['name']: bp.get('cn_name', bp['name']) for bp in risk_playbooks}

        # ▼▼▼【代码修改 V202.1】: 使用 risk_score 作为筛选标准 ▼▼▼
        # 2. 筛选出所有产生了风险分的交易日
        risk_days_df = daily_analysis_df[daily_analysis_df.get('risk_score', 0) > 0].copy()
        # ▲▲▲【代码修改 V202.1】▲▲▲
        
        if risk_days_df.empty:
            return []

        # 3. 获取风险归因的详细报告
        risk_details_df = getattr(self.tactical_engine, '_last_risk_details_df', pd.DataFrame())

        records = []
        for timestamp, row in risk_days_df.iterrows():
            # exit_signal_code 仍然从行数据中获取，因为它代表了最终的决策
            exit_code = int(row.get('exit_signal_code', 0))
            
            # 4. 找出当天具体是哪些风险剧本被触发了
            triggered_risks_en = []
            if not risk_details_df.empty and timestamp in risk_details_df.index:
                risk_details_for_day = risk_details_df.loc[timestamp]
                active_risks = risk_details_for_day[risk_details_for_day > 0].index
                triggered_risks_en = [risk.replace('RISK_SCORE_', '') for risk in active_risks]

            # 5. 翻译成中文，并组合成最终的原因描述
            triggered_risks_cn = [risk_playbook_map.get(risk, risk) for risk in triggered_risks_en]
            reason = ", ".join(triggered_risks_cn) if triggered_risks_cn else "综合风险评分超阈值"

            # 6. 根据风险代码确定严重等级
            severity_level = 1
            if exit_code >= 120: # 对应 CRITICAL
                severity_level = 3
            elif exit_code >= 80: # 对应 HIGH
                severity_level = 2
            # MEDIUM 及以下都算 1 级预警

            # 7. 使用标准化的生成器创建记录
            record = self._create_signal_record(
                stock_code=stock_code,
                trade_time=timestamp,
                timeframe='D',
                strategy_name="RISK_MANAGEMENT",
                close_price=row.get('close_D'),
                exit_signal_code=exit_code,
                exit_severity_level=severity_level,
                exit_signal_reason=reason,
                triggered_playbooks=triggered_risks_en,
                triggered_playbooks_cn=triggered_risks_cn,
                context_snapshot={'risk_score': row.get('risk_score', 0.0)}
            )
            records.append(record)
            
        return records

    def _calculate_trend_dynamics(self, df: pd.DataFrame, timeframes: List[str], ema_period: int = 34, slope_window: int = 5) -> pd.DataFrame:
        df_copy = df.copy()
        def get_slope(y):
            if len(y.dropna()) < 2: return np.nan
            x = np.arange(len(y))
            try:
                slope, _ = np.polyfit(x, y.values, 1)
                return slope
            except (np.linalg.LinAlgError, TypeError):
                return np.nan
        for tf in timeframes:
            ema_col = f'EMA_{ema_period}_{tf}'
            close_col = f'close_{tf}'
            slope_col = f'ema_slope_{tf}'
            accel_col = f'ema_accel_{tf}'
            health_col = f'trend_health_{tf}'
            if ema_col in df_copy.columns and close_col in df_copy.columns:
                df_copy[slope_col] = df_copy[ema_col].rolling(window=slope_window).apply(get_slope, raw=False)
                df_copy[accel_col] = df_copy[slope_col].rolling(window=slope_window).apply(get_slope, raw=False)
                is_above_ema = df_copy[close_col] > df_copy[ema_col]
                is_slope_positive = df_copy[slope_col] > 0
                df_copy[health_col] = is_above_ema & is_slope_positive
                df_copy[health_col].fillna(False, inplace=True)
            else:
                df_copy[health_col] = False
                df_copy[slope_col] = np.nan
                df_copy[accel_col] = np.nan
        return df_copy

    def _prepare_intraday_signals(self, all_dfs: Dict[str, pd.DataFrame], params: dict) -> pd.DataFrame:
        """
        【V6.15 新增】
        在调用战术引擎前，预处理所有需要分钟线数据的信号。
        目前主要用于计算 VWAP 支撑确认，并将其作为一个布尔列合并到日线DataFrame中。
        这遵循了“总指挥准备一切数据”的架构原则。
        """
        df_daily = all_dfs.get('D')
        if df_daily is None or df_daily.empty:
            return df_daily

        # --- VWAP 确认逻辑 ---
        vwap_params = self.tactical_engine._get_params_block(params, 'vwap_confirmation_params', {})
        # 注意：这里我们借用了 tactical_engine 的参数解析工具函数
        
        # 默认给一个False列，如果后续不满足条件则返回这个
        df_daily['cond_vwap_support'] = False

        if not vwap_params.get('enabled', False):
            return df_daily

        timeframe = vwap_params.get('timeframe', '30')
        df_intraday = all_dfs.get(timeframe)

        if df_intraday is None or df_intraday.empty:
            print(f"    - [总指挥警告] 缺少分钟线({timeframe})数据，无法计算VWAP支撑。")
            return df_daily

        vwap_col = f'VWAP_{timeframe}'
        close_col = f'close_{timeframe}'

        if vwap_col not in df_intraday.columns or close_col not in df_intraday.columns:
            print(f"    - [总指挥警告] 分钟线数据中缺少 '{vwap_col}' 或 '{close_col}'，跳过VWAP确认。")
            return df_daily

        df_intra_filtered = df_intraday[[close_col, vwap_col]].copy()
        df_intra_filtered['date'] = df_intra_filtered.index.date

        last_bar_support = df_intra_filtered.groupby('date').apply(
            lambda g: g[close_col].iloc[-1] > g[vwap_col].iloc[-1] if not g.empty else False
        )
        
        # 将计算出的VWAP支撑信号（以date为索引）映射回日线DataFrame
        # 使用 .map() 可以安全地处理日期不匹配的情况
        df_daily['cond_vwap_support'] = df_daily.index.date.map(last_bar_support).fillna(False)
        
        print(f"    - [总指挥信息] 已预处理VWAP支撑信号，发现 {df_daily['cond_vwap_support'].sum()} 个支撑日。")
        return df_daily

    def _run_intraday_resonance_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        【V117.8 协议统一版】
        - 核心重构: 废弃了对 _prepare_intraday_db_record 的调用，转而直接使用
                    self._create_signal_record()，并删除了冗余的辅助函数。
        """
        resonance_params = self.unified_config.get('strategy_params', {}).get('trend_follow', {}).get('multi_level_resonance_params', {})
        if not resonance_params.get('enabled', False): return []
        if self.daily_analysis_df is None or self.daily_analysis_df.empty: return []
        levels = resonance_params.get('levels', [])
        if not levels: return []
        trigger_tf = levels[-1]['tf']
        if trigger_tf not in all_dfs or all_dfs[trigger_tf].empty: return []
        df_aligned = all_dfs[trigger_tf].copy()
        for level in levels[:-1]:
            level_tf = level['tf']
            if level_tf in all_dfs and not all_dfs[level_tf].empty:
                df_right = all_dfs[level_tf].copy()
                rename_map = {col: f"{col}_{level_tf}" for col in df_right.columns}
                df_right.rename(columns=rename_map, inplace=True)
                df_aligned = pd.merge_asof(left=df_aligned, right=df_right, left_index=True, right_index=True, direction='backward')
            else: return []
        dynamics_timeframes = ['60', '30']
        df_aligned = self._calculate_trend_dynamics(df_aligned, dynamics_timeframes)
        daily_score_threshold = self.unified_config.get('entry_scoring_params', {}).get('score_threshold', 100)
        daily_playbook_cols = [col for col in self.daily_analysis_df.columns if col.startswith('playbook_')]
        daily_context_cols_to_merge = ['context_mid_term_bullish', 'entry_score'] + daily_playbook_cols
        daily_context_df = self.daily_analysis_df[daily_context_cols_to_merge].copy()
        is_bullish_trend = daily_context_df['context_mid_term_bullish']
        is_reversal_day = daily_context_df['entry_score'] >= daily_score_threshold
        daily_context_df['is_daily_trend_ok'] = is_bullish_trend | is_reversal_day
        daily_context_df.rename(columns={'entry_score': 'daily_entry_score'}, inplace=True)
        df_aligned = pd.merge_asof(left=df_aligned, right=daily_context_df, left_index=True, right_index=True, direction='backward')
        df_aligned['is_daily_trend_ok'].fillna(False, inplace=True)
        df_aligned['daily_entry_score'].fillna(0, inplace=True)
        for col in daily_playbook_cols:
            if col in df_aligned.columns: df_aligned[col].fillna(False, inplace=True)
        final_signal = pd.Series(True, index=df_aligned.index)
        final_signal &= df_aligned['is_daily_trend_ok']
        final_signal &= df_aligned.get('trend_health_60', False)
        final_signal &= df_aligned.get('trend_health_30', False)
        final_signal &= (df_aligned.get('ema_accel_30', 0) >= 0)
        if final_signal.sum() == 0: return []
        for i, level in enumerate(levels):
            level_tf, level_logic, level_conditions = level['tf'], level.get('logic', 'AND').upper(), level.get('conditions', [])
            level_signal = pd.Series(True if level_logic == 'AND' else False, index=df_aligned.index)
            for cond in level_conditions:
                cond_signal = self._check_single_condition(df_aligned, cond, level_tf)
                if level_logic == 'AND': level_signal &= cond_signal
                else: level_signal |= cond_signal
            final_signal &= level_signal
        triggered_df = df_aligned[final_signal]
        if triggered_df.empty: return []
        db_records = []
        for timestamp, row in triggered_df.iterrows():
            resonance_playbook = resonance_params.get('signal_name', 'UNKNOWN_RESONANCE')
            daily_playbooks = [col.replace('playbook_', '') for col in row.index if col.startswith('playbook_') and row[col] is True]
            combined_playbooks = list(set([resonance_playbook] + daily_playbooks))
            daily_score = row.get('daily_entry_score', 0.0)
            resonance_score = resonance_params.get('score', 0.0)
            total_score = daily_score + resonance_score
            
            # ▼▼▼【代码修改 V117.8】: 直接调用标准生成器，不再使用 _prepare_intraday_db_record ▼▼▼
            record = self._create_signal_record(
                stock_code=stock_code,
                trade_time=timestamp,
                timeframe=trigger_tf,
                strategy_name=resonance_params.get('signal_name', 'UNKNOWN_RESONANCE'),
                close_price=row.get('close'),
                entry_score=total_score,
                entry_signal=True,
                triggered_playbooks=combined_playbooks,
                context_snapshot={'close': row.get('close'), 'daily_score': daily_score, 'resonance_score': resonance_score}
            )
            # ▲▲▲【代码修改 V117.8】▲▲▲
            db_records.append(record)
        return db_records

    # ▼▼▼【代码修改】: 止盈引擎重构，实现三级警报系统 ▼▼▼
    def _run_intraday_take_profit_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        【V117.8 协议统一版】
        - 核心重构: 废弃了手动创建字典的方式，转而调用 self._create_signal_record() 辅助函数
                    来生成止盈信号记录，确保了战报格式的绝对统一。
        """
        tp_params = self.unified_config.get('strategy_params', {}).get('trend_follow', {}).get('intraday_take_profit_params', {})
        if not tp_params.get('enabled', False): return []
        
        tf = tp_params.get('timeframe')
        if not tf or tf not in all_dfs or all_dfs[tf].empty: return []
        
        df = all_dfs[tf].copy()

        dynamics_timeframes = ['60', '30']
        for health_tf in dynamics_timeframes:
            if health_tf in all_dfs and not all_dfs[health_tf].empty:
                df_right = all_dfs[health_tf].copy()
                rename_map = {col: f"{col}_{health_tf}" for col in df_right.columns}
                df_right.rename(columns=rename_map, inplace=True)
                df = pd.merge_asof(left=df, right=df_right, left_index=True, right_index=True, direction='backward')
        
        daily_support_ma = 'EMA_55_D'
        if 'D' in all_dfs and daily_support_ma in all_dfs['D'].columns:
            df = pd.merge_asof(left=df, right=all_dfs['D'][[daily_support_ma]], left_index=True, right_index=True, direction='backward')

        df = self._calculate_trend_dynamics(df, dynamics_timeframes, ema_period=34, slope_window=5)

        signals = []
        
        is_still_rising = df.get('ema_slope_30', 0) > 0
        is_decelerating = df.get('ema_accel_30', 0) < 0
        was_accelerating = df.get('ema_accel_30', 0).shift(1) >= 0
        level_1_signal = is_still_rising & is_decelerating & was_accelerating
        if level_1_signal.any():
            signals.append({'level': 1, 'reason': '30分钟趋势加速度转负', 'signal': level_1_signal})

        p = [12, 26, 9]
        macd_col, signal_col = f'MACD_{p[0]}_{p[1]}_{p[2]}', f'MACDs_{p[0]}_{p[1]}_{p[2]}'
        if macd_col in df.columns and signal_col in df.columns:
            base_signal = (df[macd_col] < df[signal_col]) & (df[macd_col].shift(1) >= df[signal_col].shift(1))
            is_trend_deteriorating = df.get('trend_health_30', False) == False
            level_2_signal = base_signal & is_trend_deteriorating
            if level_2_signal.any():
                signals.append({'level': 2, 'reason': f'{tf}分钟MACD死叉且30分钟趋势不健康', 'signal': level_2_signal})

        if daily_support_ma in df.columns:
            level_3_signal = (df['close'] < df[daily_support_ma]) & (df['close'].shift(1) >= df[daily_support_ma].shift(1))
            if level_3_signal.any():
                signals.append({'level': 3, 'reason': f'价格跌破日线关键支撑({daily_support_ma})', 'signal': level_3_signal})

        if not signals: return []
        
        df['exit_severity_level'] = 0
        df['exit_signal_reason'] = ''
        
        for s in sorted(signals, key=lambda x: x['level'], reverse=True):
            df.loc[s['signal'], 'exit_severity_level'] = s['level']
            df.loc[s['signal'], 'exit_signal_reason'] = s['reason']
            
        triggered_df = df[df['exit_severity_level'] > 0].copy()
        if triggered_df.empty: return []

        db_records = []
        for timestamp, row in triggered_df.iterrows():
            # ▼▼▼【代码修改 V117.8】: 使用新的“标准战报生成器”创建记录 ▼▼▼
            record = self._create_signal_record(
                stock_code=stock_code,
                trade_time=timestamp,
                timeframe=tf,
                strategy_name=tp_params.get('signal_name', 'INTRADAY_TAKE_PROFIT'),
                close_price=row.get('close'),
                exit_signal_code=100 + int(row.get('exit_severity_level', 0)),
                exit_severity_level=row.get('exit_severity_level'),
                exit_signal_reason=row.get('exit_signal_reason'),
                triggered_playbooks=[f"EXIT_LEVEL_{int(row.get('exit_severity_level', 0))}"],
                context_snapshot={'close': row.get('close'), 'reason': row.get('exit_signal_reason')},
            )
            # ▲▲▲【代码修改 V117.8】▲▲▲
            db_records.append(record)
        return db_records

    def _check_single_condition(self, df: pd.DataFrame, cond: Dict, tf: str) -> pd.Series:
        # ... (此函数保持不变) ...
        cond_type = cond['type']
        resonance_config = self.unified_config.get('strategy_params', {}).get('trend_follow', {}).get('multi_level_resonance_params', {})
        trigger_tf_str = resonance_config.get('levels', [{}])[-1].get('tf')
        suffix = f'_{tf}' if tf != trigger_tf_str else ''
        try:
            trigger_minutes = int(trigger_tf_str)
            condition_minutes = int(tf)
            shift_periods = max(1, condition_minutes // trigger_minutes)
        except (ValueError, ZeroDivisionError):
            shift_periods = 1
        def check_cols(*cols):
            missing_cols = [col for col in cols if col not in df.columns]
            if missing_cols:
                return False
            return True
        if cond_type == 'ema_above':
            period = cond['period']
            ema_col, close_col = f'EMA_{period}{suffix}', f'close{suffix}'
            if check_cols(ema_col, close_col): return df[close_col] > df[ema_col]
        elif cond_type == 'macd_above_zero':
            p = cond['periods']
            macd_line_col = f'MACD_{p[0]}_{p[1]}_{p[2]}{suffix}'
            hist_col = f'MACDh_{p[0]}_{p[1]}_{p[2]}{suffix}'
            if check_cols(macd_line_col, hist_col):
                is_above_zero_and_rising = (df[macd_line_col] > 0) & (df[macd_line_col] > df[macd_line_col].shift(shift_periods))
                hist_above_zero_strengthening = (df[hist_col] > 0) & (df[hist_col] > df[hist_col].shift(shift_periods)) & \
                                                (df[hist_col].shift(shift_periods * 2) < df[hist_col].shift(shift_periods))
                return is_above_zero_and_rising | hist_above_zero_strengthening
        elif cond_type == 'macd_cross':
            p = cond['periods']
            hist_col = f'MACDh_{p[0]}_{p[1]}_{p[2]}{suffix}'
            if check_cols(hist_col): return (df[hist_col] > 0) & (df[hist_col].shift(shift_periods) <= 0)
        elif cond_type == 'macd_hist_turning_up':
            p = cond['periods']
            hist_col = f'MACDh_{p[0]}_{p[1]}_{p[2]}{suffix}'
            if check_cols(hist_col): return df[hist_col] > df[hist_col].shift(shift_periods)
        elif cond_type == 'dmi_cross':
            p = cond['period']
            pdi_col, mdi_col = f'DMP_{p}{suffix}', f'DMN_{p}{suffix}'
            if check_cols(pdi_col, mdi_col): return (df[pdi_col] > df[mdi_col]) & (df[pdi_col].shift(shift_periods) <= df[mdi_col].shift(shift_periods))
        elif cond_type == 'kdj_cross':
            p = cond['periods']
            k_col, d_col = f'KDJk_{p[0]}_{p[1]}_{p[2]}{suffix}', f'KDJd_{p[0]}_{p[1]}_{p[2]}{suffix}'
            oversold_level = cond.get('low_level', 50)
            if check_cols(k_col, d_col):
                is_cross = (df[k_col] > df[d_col]) & (df[k_col].shift(shift_periods) <= df[d_col].shift(shift_periods))
                is_in_zone = df[d_col] < oversold_level
                return is_cross & is_in_zone
        elif cond_type == 'kdj_j_reversal':
            p = cond['periods']
            j_col = f'KDJj_{p[0]}_{p[1]}_{p[2]}{suffix}'
            low_level = cond.get('low_level', 30)
            if check_cols(j_col):
                is_turning_up = (df[j_col] > df[j_col].shift(shift_periods))
                was_in_low_zone = (df[j_col].shift(shift_periods) < low_level)
                return is_turning_up & was_in_low_zone
        elif cond_type == 'rsi_reversal':
            p = cond['period']
            rsi_col = f'RSI_{p}{suffix}'
            oversold_level = cond.get('oversold_level', 35)
            if check_cols(rsi_col):
                classic_reversal = (df[rsi_col] > oversold_level) & (df[rsi_col].shift(shift_periods) <= oversold_level)
                is_turning_up_after_dip = (df[rsi_col] > df[rsi_col].shift(shift_periods)) & \
                                          (df[rsi_col].shift(shift_periods) < df[rsi_col].shift(shift_periods * 2))
                return classic_reversal | is_turning_up_after_dip
        return pd.Series(False, index=df.index)

    async def debug_run_for_period(self, stock_code: str, start_date: str, end_date: str):
        """
        【V202.0 终极战报适配版】
        - 核心升级: 彻底重构报告展示逻辑，使其能够按优先级正确识别并清晰地展示
                    所有类型的信号：买入信号、风险/出场信号、以及持仓管理动作。
        """
        print("=" * 80)
        print(f"--- [历史回溯调试启动 (V202.0 终极战报适配版)] ---")
        print(f"    -> 股票代码: {stock_code}")
        print(f"    -> 回测时段: {start_date} to {end_date}")
        print("=" * 80)

        try:
            print(f"\n[步骤 1/3] 正在调用总指挥部 (run_for_stock) 获取完整信号列表...")
            all_records = await self.run_for_stock(stock_code, trade_time=end_date)
            
            if all_records is None:
                print("[错误] 总指挥部未能返回任何信号记录。")
                return

            print(f"[成功] 总指挥部运行完毕，共返回 {len(all_records)} 条原始信号记录。")

            print("\n[步骤 2/3] [调试专属] 正在执行波段跟踪模拟器...")
            print("====== 【波段跟踪模拟器 V85.2】执行完毕 ======")

            print(f"\n[步骤 3/3] 正在筛选并展示目标时段 ({start_date} to {end_date}) 的所有信号...")
            
            start_dt = pd.to_datetime(start_date, utc=True)
            end_dt = pd.to_datetime(end_date, utc=True).replace(hour=23, minute=59, second=59)
            
            debug_period_records = []
            for rec in all_records:
                # 确保时间是时区感知的UTC时间
                rec_time = pd.to_datetime(rec['trade_time'])
                if rec_time.tzinfo is None:
                    rec_time = rec_time.tz_localize('UTC')
                else:
                    rec_time = rec_time.tz_convert('UTC')

                if start_dt <= rec_time <= end_dt:
                    debug_period_records.append(rec)

            if not debug_period_records:
                print(f"[信息] 在指定时段 {start_date} to {end_date} 内没有找到任何信号。")
                print("--- [历史回溯调试完成] ---")
                print("=" * 80)
                return

            # 按时间排序，确保战报的连贯性
            debug_period_records.sort(key=lambda x: pd.to_datetime(x['trade_time'], utc=True))

            print("\n" + "="*30 + " [全流程信号透视报告] " + "="*30)
            
            # ▼▼▼【代码修改 V202.0】: 全新的、按优先级解析的战报展示逻辑 ▼▼▼
            for record in debug_period_records:
                time_obj = pd.to_datetime(record['trade_time'])
                time_str = time_obj.strftime('%Y-%m-%d %H:%M:%S %Z')
                tf = record.get('timeframe', 'N/A')
                
                signal_type = "未知信号"
                details = "无详细信息"

                # 优先级1: 入场信号
                if record.get('entry_signal'):
                    score = record.get('entry_score', 0.0)
                    # 优先使用中文剧本名，如果不存在则使用英文名
                    playbooks = record.get('triggered_playbooks_cn', record.get('triggered_playbooks', []))
                    # 过滤掉可能存在的空字符串或None
                    playbooks = [p for p in playbooks if p]
                    signal_type = "买入信号"
                    details = f"得分: {score:<7.2f} | 剧本: {', '.join(playbooks)}"
                
                # 优先级2: 风险/出场信号
                elif record.get('exit_signal_code', 0) > 0:
                    severity = record.get('exit_severity_level', 0)
                    # 优先使用中文原因，如果不存在则使用英文原因
                    reason_list_cn = record.get('triggered_playbooks_cn', [])
                    reason_list_en = record.get('triggered_playbooks', [])
                    reason_list = reason_list_cn if any(reason_list_cn) else reason_list_en
                    reason = ", ".join(reason_list) if reason_list else record.get('exit_signal_reason', 'N/A')
                    
                    signal_type = f"卖出警报(L{severity})"
                    details = f"原因: {reason}"

                # 优先级3: 持仓管理动作
                elif 'trade_action' in record and record.get('trade_action'):
                    action = record['trade_action']
                    reason = record.get('alert_reason', 'N/A')
                    position_size = record.get('position_size', 0.0)
                    signal_type = "战术动作"
                    details = f"动作: {action} | 原因: {reason} | 剩余仓位: {position_size:.0%}"
                
                # 打印最终格式化的战报
                print(f"{time_str}  [周期:{tf:>3s}] [类型:{signal_type:<12s}] | {details}")
            # ▲▲▲【代码修改 V202.0】▲▲▲

            print(f"--- [历史回溯调试完成] ---")
            print("=" * 80)

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
