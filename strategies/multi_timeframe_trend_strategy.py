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
from strategies.weekly_trend_follow_strategy import WeeklyTrendFollowStrategy
from utils.config_loader import load_strategy_config
from utils.data_sanitizer import sanitize_for_json

logger = logging.getLogger(__name__)

class MultiTimeframeTrendStrategy:

    def __init__(self):
        """
        【V6.7 分级止盈系统】
        - 核心升级: 建立三级止盈预警系统，提供差异化、可操作的卖出建议。
          - 一级预警 (黄色): 趋势加速度转负，提示“关注或部分减仓”。
          - 二级警报 (橙色): 短期指标死叉，提示“标准止盈”。
          - 三级警报 (红色): 跌破日线关键支撑，提示“紧急离场”。
        - 优化: 信号记录中增加 `exit_severity_level` 和 `exit_signal_reason` 字段，分析报告更精细。
        """
        tactical_config_path = 'config/trend_follow_strategy.json'
        strategic_config_path = 'config/weekly_trend_follow_strategy.json'
        self.tactical_config = load_strategy_config(tactical_config_path)
        self.strategic_config = load_strategy_config(strategic_config_path)
        
        # ▼▼▼ 调整配置合并逻辑，确保所有参数块都被包含 ▼▼▼
        # 1. 深度复制战术配置作为基础
        self.merged_config = deepcopy(self.tactical_config)

        # 2. 合并特征工程参数 (feature_engineering_params)
        base_merged_fe_params = self._merge_feature_engineering_configs(
            self.tactical_config.get('feature_engineering_params', {}),
            self.strategic_config.get('feature_engineering_params', {})
        )
        # 发现并合并共振和止盈所需的额外指标
        resonance_indicators = self._discover_resonance_indicators(self.tactical_config)
        take_profit_indicators = self._discover_take_profit_indicators(self.tactical_config)
        temp_indicators = self._merge_indicators(base_merged_fe_params.get('indicators', {}), resonance_indicators)
        final_indicators = self._merge_indicators(temp_indicators, take_profit_indicators)
        base_merged_fe_params['indicators'] = final_indicators
        self.merged_config['feature_engineering_params'] = base_merged_fe_params

        # 3. 合并战略引擎的剧本 (如果存在)
        if 'strategy_playbooks' in self.strategic_config:
            self.merged_config['strategy_playbooks'] = deepcopy(self.strategic_config['strategy_playbooks'])
        
        # 4. 确保战术引擎的参数块 (如 chip_feature_params) 也被正确合并
        #    这一步通常在第一步的 deepcopy 中已经完成，但为了明确，可以再次检查
        if 'strategy_params' in self.tactical_config:
            self.merged_config['strategy_params'] = deepcopy(self.tactical_config['strategy_params'])

        self.indicator_service = IndicatorService()
        self.strategic_engine = WeeklyTrendFollowStrategy(config=self.strategic_config) 
        self.tactical_engine = TrendFollowStrategy(config=self.tactical_config)
        self.daily_analysis_df = None
        self.required_timeframes = self.indicator_service._discover_required_timeframes_from_config(self.merged_config)

    # ▼▼▼【 “军用标准战报”生成器 ▼▼▼
    def _create_signal_record(self, **kwargs) -> Dict[str, Any]:
        """
        【V117.15 时区校准版】标准信号记录生成器。
        - 核心修复: 强制将所有 trade_time 转换为带时区的 UTC 时间。
        - 工作流程:
          1. 使用 pd.to_datetime 将任何输入转为 pandas Timestamp。
          2. 如果时间戳是“天真”的(naive)，则假定它是本地时间('Asia/Shanghai')并进行“本地化”。
          3. 最终，使用 .tz_convert('UTC') 将其统一转换为UTC时间。
          4. 调用 .to_pydatetime() 得到一个标准的、带时区的Python datetime对象。
        - 收益: 从根源上统一了整个系统的时间标准，确保与数据库的UTC存储完全兼容。
        """
        trade_time_input = kwargs.get('trade_time')
        if trade_time_input is None:
            raise ValueError("创建信号记录时必须提供 'trade_time'")
        
        # ▼▼▼ 强制时区校准为UTC ▼▼▼
        ts = pd.to_datetime(trade_time_input)
        # 检查时间是否是“天真”的
        if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
            # 如果是天真的，假定为本地时间并进行本地化，然后转为UTC
            standard_trade_time = ts.tz_localize('Asia/Shanghai').tz_convert('UTC').to_pydatetime()
            # print(f"调试信息: [create_signal] 发现天真时间 {ts}, 已转换为UTC: {standard_trade_time}")
        else:
            # 如果已经带时区，直接转换为UTC
            standard_trade_time = ts.tz_convert('UTC').to_pydatetime()
            # print(f"调试信息: [create_signal] 发现带时区时间 {ts}, 已转换为UTC: {standard_trade_time}")

        record = {
            "stock_code": None,
            "trade_time": standard_trade_time, # 现在这里永远是UTC时间
            "timeframe": "N/A",
            "strategy_name": "UNKNOWN",
            "close_price": 0.0,
            "entry_score": 0.0,
            "entry_signal": False,
            "exit_signal_code": 0,
            "exit_severity_level": 0,
            "exit_signal_reason": None,
            "triggered_playbooks": [],
            "context_snapshot": {},
            "analysis_text": None
        }
        
        record.update(kwargs)
        
        record['close_price'] = sanitize_for_json(record['close_price'])
        record['context_snapshot'] = sanitize_for_json(record['context_snapshot'])

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

    # ... (从 _discover_take_profit_indicators 到 _run_tactical_engine 的所有函数保持不变) ...
    def _discover_take_profit_indicators(self, config: Dict) -> Dict:
        discovered = defaultdict(lambda: {'enabled': True, 'configs': []})
        tp_params = config.get('strategy_params', {}).get('trend_follow', {}).get('intraday_take_profit_params', {})
        if not tp_params.get('enabled', False):
            return {}
        tf = tp_params.get('timeframe')
        if not tf:
            return {}
        for rule in tp_params.get('rules', []):
            rule_type = rule.get('type')
            indicator_name, params = None, None
            if rule_type == 'macd_dead_cross':
                indicator_name, params = 'macd', {'apply_on': [tf], 'periods': rule['periods']}
            elif rule_type == 'kdj_dead_cross':
                indicator_name, params = 'kdj', {'apply_on': [tf], 'periods': rule['periods']}
            elif rule_type == 'top_divergence' and rule.get('indicator') == 'rsi':
                indicator_name, params = 'rsi', {'apply_on': [tf], 'periods': [rule['periods']]}
            if indicator_name and params and params not in discovered[indicator_name]['configs']:
                discovered[indicator_name]['configs'].append(params)
        return json.loads(json.dumps(discovered))

    def _discover_resonance_indicators(self, config: Dict) -> Dict:
        discovered = defaultdict(lambda: {'enabled': True, 'configs': []})
        resonance_params = config.get('strategy_params', {}).get('trend_follow', {}).get('multi_level_resonance_params', {})
        if not resonance_params.get('enabled', False): return {}
        for level in resonance_params.get('levels', []):
            tf = level['tf']
            for cond in level.get('conditions', []):
                cond_type, params, indicator_name = cond['type'], None, None
                if cond_type in ('macd_above_zero', 'macd_cross', 'macd_hist_turning_up'):
                    indicator_name, params = 'macd', {'apply_on': [tf], 'periods': cond['periods']}
                elif cond_type == 'dmi_cross':
                    indicator_name, params = 'dmi', {'apply_on': [tf], 'periods': [cond['period']]}
                elif cond_type == 'kdj_cross':
                    indicator_name, params = 'kdj', {'apply_on': [tf], 'periods': cond['periods']}
                elif cond_type == 'rsi_reversal':
                    indicator_name, params = 'rsi', {'apply_on': [tf], 'periods': [cond['period']]}
                elif cond_type == 'ema_above':
                    indicator_name, params = 'ema', {'apply_on': [tf], 'periods': [cond['period']]}
                if indicator_name and params and params not in discovered[indicator_name]['configs']:
                    discovered[indicator_name]['configs'].append(params)
        return json.loads(json.dumps(discovered))

    def _merge_feature_engineering_configs(self, tactical_fe, strategic_fe):
        merged = deepcopy(tactical_fe)
        merged['base_needed_bars'] = max(
            tactical_fe.get('base_needed_bars', 0),
            strategic_fe.get('base_needed_bars', 0)
        )
        merged['indicators'] = self._merge_indicators(
            tactical_fe.get('indicators', {}),
            strategic_fe.get('indicators', {})
        )
        return merged

    def _merge_indicators(self, base_indicators, new_indicators):
        merged = deepcopy(base_indicators)
        all_keys = set(merged.keys()) | set(new_indicators.keys())
        def standardize_to_configs(cfg):
            if not cfg or not cfg.get('enabled', False): return []
            if 'configs' in cfg: return deepcopy(cfg['configs'])
            if 'apply_on' in cfg:
                sub_cfg = {'apply_on': cfg['apply_on']}
                if 'periods' in cfg: sub_cfg['periods'] = cfg['periods']
                if 'std_dev' in cfg: sub_cfg['std_dev'] = cfg['std_dev']
                return [sub_cfg]
            return []
        for key in all_keys:
            if key == '说明': continue
            base_cfg, new_cfg = merged.get(key, {}), new_indicators.get(key, {})
            is_enabled = base_cfg.get('enabled', False) or new_cfg.get('enabled', False)
            if not is_enabled: continue
            base_sub_configs, new_sub_configs = standardize_to_configs(base_cfg), standardize_to_configs(new_cfg)
            final_configs = base_sub_configs
            for sub_cfg in new_sub_configs:
                if sub_cfg not in final_configs: final_configs.append(sub_cfg)
            if not final_configs:
                if key in base_cfg or key in new_cfg:
                     merged[key] = deepcopy(base_cfg); merged[key].update(deepcopy(new_cfg))
                continue
            merged[key] = {
                'enabled': True,
                '说明': base_cfg.get('说明', '') or new_cfg.get('说明', ''),
                'configs': final_configs
            }
            if not final_configs and 'enabled' in (base_cfg or new_cfg):
                 merged[key] = {'enabled': is_enabled, '说明': merged[key]['说明']}
        return merged

    async def run_for_stock(self, stock_code: str, trade_time: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        【V117.9 链路修复版】
        - 核心修复: 增加了 return 语句。函数现在会收集所有五个引擎生成的信号记录，
                    进行合并与去重，并最终返回一个统一的列表。这打通了策略执行与数据
                    库保存之间的“情报断链”。
        """
        logger.info(f"--- 开始为【{stock_code}】执行五级引擎分析 (V117.9) ---")
        
        # --- 准备阶段 ---
        logger.info(f"--- 准备阶段: 调用 IndicatorService 统一准备所有数据... ---")
        all_dfs = await self.indicator_service._prepare_base_data_and_indicators(stock_code, self.merged_config, trade_time)
        if 'D' not in all_dfs or 'W' not in all_dfs:
            logger.warning(f"[{stock_code}] 核心数据(周线或日线)准备失败，分析终止。")
            return None
            
        # --- 引擎1: 战略引擎 (周线) ---
        logger.info(f"\n--- 引擎1: 开始运行【战略引擎】(周线)... ---")
        strategic_signals_df = self._run_strategic_engine(all_dfs['W'])
        logger.info(f"--- 引擎1: 【战略引擎】运行完毕。---")
        
        # --- 数据流转: 战略信号注入日线 ---
        logger.info(f"\n--- 数据流转: 整合战略信号到日线数据... ---")
        all_dfs['D'] = self._merge_strategic_signals_to_daily(all_dfs['D'], strategic_signals_df)
        
        # --- 引擎2: 战术引擎 (日线) ---
        logger.info(f"\n--- 引擎2: 开始运行【战术引擎】(日线)... ---")
        tactical_records = self._run_tactical_engine(stock_code, all_dfs)
        logger.info(f"--- 引擎2: 【战术引擎】运行完毕，生成 {len(tactical_records)} 条日线信号。 ---")
        
        # --- 引擎3: 执行引擎-买入 (分钟线共振) ---
        logger.info(f"\n--- 引擎3: 开始运行【执行引擎-买入】(分钟线)... ---")
        resonance_entry_records = self._run_intraday_resonance_engine(stock_code, all_dfs)
        logger.info(f"--- 引擎3: 【执行引擎-买入】运行完毕，生成 {len(resonance_entry_records)} 条分钟线买入信号。 ---")
        
        # --- 引擎4: 通用盘中预警引擎 (分钟线风险) ---
        logger.info(f"\n--- 引擎4: 开始运行【通用盘中预警引擎】(分钟线)... ---")
        risk_alert_records = self._run_intraday_alert_engine(stock_code, all_dfs)
        logger.info(f"--- 引擎4: 【通用盘中预警引擎】运行完毕，生成 {len(risk_alert_records)} 条分钟线风险警报。 ---")

        # --- 引擎5: 通用盘中买入确认引擎 (分钟线) ---
        logger.info(f"\n--- 引擎5: 开始运行【通用盘中买入确认引擎】(分钟线)... ---")
        confirmation_entry_records = self._run_intraday_entry_engine(stock_code, all_dfs)
        logger.info(f"--- 引擎5: 【通用盘中买入确认引擎】运行完毕，生成 {len(confirmation_entry_records)} 条分钟线买入确认信号。 ---")

        # --- 最终阶段: 合并所有信号 ---
        logger.info("\n--- 最终阶段: 合并所有信号记录... ---")
        
        # 将所有买入信号（日线、分钟共振、分钟确认）合并
        all_entry_records = tactical_records + resonance_entry_records + confirmation_entry_records
        
        # 注意：这里的去重逻辑现在变得至关重要，但我们之前的视图层重构已经解决了这个问题。
        # 视图层会从所有信号中找到最新的那个。所以这里的简单合并是安全的。
        
        # 将所有买入信号与所有风险信号合并
        all_records = all_entry_records + risk_alert_records
        
        logger.info(f"--- 所有引擎分析完毕，共生成 {len(all_records)} 条最终信号记录准备交付。 ---")
        
        return all_records

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
            pd.to_datetime(rec['trade_time']).date(): rec
            for rec in daily_records if rec.get('entry_signal')
        }
        
        fused_dates = set()

        for conf_rec in confirmation_records:
            conf_date = pd.to_datetime(conf_rec['trade_time']).date()
            
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

    def _run_intraday_entry_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        【V117.37 架构优化版】
        - 核心升级: 调用 TradeCalendar.get_next_trade_date() 来获取监控目标日，取代了之前在DataFrame中查找的方式。
        - 收益:
          1. 逻辑更清晰，责任更明确，符合Django最佳实践。
          2. 更健壮，不再依赖 all_dfs['D'] 的完整性，即使日线数据有缺失也能正确找到下一个交易日。
          3. 性能可能更高，因为数据库查询经过优化，通常比在大型DataFrame中定位要快。
        """
        print("--- [引擎5-调试 V117.37] 进入分钟买入确认引擎 (DAO增强版) ---")
        all_confirmation_records = []
        entry_params = self.tactical_engine._get_params_block(self.tactical_config, 'intraday_entry_params', {})
        get_val = self.tactical_engine._get_param_value
        
        is_enabled = get_val(entry_params.get('enabled'), False)
        if not is_enabled: return all_confirmation_records
        
        if self.daily_analysis_df is None or self.daily_analysis_df.empty: return all_confirmation_records

        daily_score_threshold = get_val(entry_params.get('daily_score_threshold'), 100)
        setup_days_df = self.daily_analysis_df[self.daily_analysis_df['entry_score'] >= daily_score_threshold]
        
        if setup_days_df.empty: return all_confirmation_records
            
        minute_tf = str(get_val(entry_params.get('timeframe'), '5'))
        minute_df = all_dfs.get(minute_tf)
        if minute_df is None or minute_df.empty: return all_confirmation_records
        
        rules = entry_params.get('confirmation_rules', {})
        min_time_after_open = get_val(rules.get('min_time_after_open'), 15)

        print(f"--- [引擎5-调试] 发现 {len(setup_days_df)} 个日线高分预备日，将启动次日盘中监控...")
        for setup_date_ts, setup_row in setup_days_df.iterrows():
            # setup_date_ts 是一个带时区的 pandas.Timestamp
            setup_date = setup_date_ts.date() # 提取日期部分

            # ▼▼▼【代码修改 V117.37】: 调用新式武器获取下一个交易日 ▼▼▼
            monitoring_date = TradeCalendar.get_next_trade_date(reference_date=setup_date)
            
            if monitoring_date is None:
                print(f"\n--- [引擎5-调试] 预备日 {setup_date} 是最后一个已知交易日，无法监控次日，跳过。")
                continue
            # ▲▲▲【代码修改 V117.37】▲▲▲

            print(f"\n--- [引擎5-调试] 预备日: {setup_date} (分数: {setup_row.get('entry_score', 0):.0f}) -> 监控目标日: {monitoring_date} ---")
            
            # 使用监控目标日来筛选分钟线数据
            # 注意：minute_df.index.date 是 date 对象，monitoring_date 也是 date 对象，可以直接比较
            alert_day_minute_df = minute_df[minute_df.index.date == monitoring_date].copy()
            
            if alert_day_minute_df.empty:
                print(f"    - [调试] 警告: 未找到监控目标日 {monitoring_date} 的分钟线数据，跳过。")
                continue
            print(f"    - [调试] 已提取目标日分钟线数据 {len(alert_day_minute_df)} 条。")

            final_confirmation_signal = pd.Series(True, index=alert_day_minute_df.index)
            close_col_m = f'close_{minute_tf}'
            
            vwap_rule = rules.get('vwap_reclaim', {})
            if get_val(vwap_rule.get('enabled'), False):
                vwap_col_m = f'VWAP_{minute_tf}'
                if vwap_col_m in alert_day_minute_df.columns and close_col_m in alert_day_minute_df.columns:
                    vwap_signal = (alert_day_minute_df[close_col_m] > alert_day_minute_df[vwap_col_m])
                    print(f"    - [调试-规则] VWAP突破: 触发了 {vwap_signal.sum()} 次。")
                    final_confirmation_signal &= vwap_signal
                else:
                    print(f"    - [调试-规则] VWAP突破: 缺少列 {vwap_col_m}，规则失效。")
                    final_confirmation_signal &= False

            vol_rule = rules.get('volume_confirmation', {})
            if get_val(vol_rule.get('enabled'), False):
                volume_col_m = f'volume_{minute_tf}'
                vol_ma_period = get_val(vol_rule.get('ma_period'), 21)
                vol_ma_col_m = f'VOL_MA_{vol_ma_period}_{minute_tf}'
                if vol_ma_col_m in alert_day_minute_df.columns and volume_col_m in alert_day_minute_df.columns:
                    vol_signal = (alert_day_minute_df[volume_col_m] > alert_day_minute_df[vol_ma_col_m])
                    print(f"    - [调试-规则] 成交量确认: 触发了 {vol_signal.sum()} 次。")
                    final_confirmation_signal &= vol_signal
                else:
                    print(f"    - [调试-规则] 成交量确认: 缺少列 {vol_ma_col_m}，规则失效。")
                    final_confirmation_signal &= False

            print(f"    - [调试] 所有规则叠加后，共有 {final_confirmation_signal.sum()} 个K线满足静态条件。")
            
            # ▼▼▼【代码修改 V117.29】: 修正监控开始时间的时区问题 ▼▼▼
            # 1. 从带时区的 setup_date (UTC) 中获取本地日期对象
            local_date = setup_date.tz_convert('Asia/Shanghai').date()
            
            # 2. 使用本地日期创建“天真”的本地开盘时间
            naive_market_open_time = datetime.combine(local_date, time(9, 30))
            
            # 3. 将“天真”时间本地化为正确的时区，使其成为“感知”时间
            aware_market_open_time = pd.Timestamp(naive_market_open_time, tz='Asia/Shanghai')
            
            # 4. 计算最终的监控开始时间
            monitoring_start_time = aware_market_open_time + pd.Timedelta(minutes=min_time_after_open)
            print(f"    - [调试] 监控起始时间已校准为: {monitoring_start_time}")
            # ▲▲▲【代码修改 V117.29】▲▲▲

            triggered_minutes = alert_day_minute_df[
                (alert_day_minute_df.index >= monitoring_start_time) &
                (final_confirmation_signal == True)
            ]
            
            print(f"    - [调试] 在 {monitoring_start_time.time()} 之后，找到 {len(triggered_minutes)} 个满足条件的K线。")

            if not triggered_minutes.empty:
                first_confirmation_minute = triggered_minutes.iloc[0]
                confirm_time = first_confirmation_minute.name
                confirm_price = first_confirmation_minute[f'close_{minute_tf}']
                daily_score = setup_row.get('entry_score', 0)
                bonus_score = get_val(entry_params.get('bonus_score'), 50)
                final_score = daily_score + bonus_score
                daily_playbooks = [p.replace('playbook_', '') for p in setup_row.index if p.startswith('playbook_') and setup_row[p] is True]
                record = self._create_signal_record(
                    stock_code=stock_code,
                    trade_time=confirm_time,
                    timeframe=minute_tf,
                    strategy_name=get_val(entry_params.get('strategy_name'), 'INTRADAY_ENTRY_CONFIRMATION'),
                    close_price=confirm_price,
                    entry_score=final_score,
                    entry_signal=True,
                    triggered_playbooks=list(set(daily_playbooks + [get_val(entry_params.get('playbook_name'), 'ENTRY_INTRADAY_CONFIRMATION')])),
                    context_snapshot={'close': confirm_price, 'daily_score': daily_score, 'bonus': bonus_score, 'intraday_confirmed': True},
                )
                all_confirmation_records.append(record)
                print(f"    - [信号生成!] 已在 {confirm_time.time()} 生成分钟线买入信号，最终得分: {final_score:.0f}")
                continue 
        
        print("--- [引擎5-调试] 分钟买入确认引擎执行完毕 ---")
        return all_confirmation_records

    def _run_intraday_alert_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        【V117.8 协议统一版】
        - 核心重构: 废弃了手动创建字典的方式，转而调用 self._create_signal_record() 辅助函数
                    来生成信号记录，确保了战报格式的绝对统一。
        """
        all_alerts = []
        exec_params = self.tactical_engine._get_params_block(self.tactical_config, 'intraday_execution_params', {})
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

        upthrust_calc_params = self.tactical_engine._get_params_block(self.tactical_config, 'exit_strategy_params', {}).get('upthrust_distribution_params', {})
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

            triggered_minutes = alert_day_minute_df[
                (alert_day_minute_df[close_col] < alert_day_minute_df[vwap_col]) &
                (alert_day_minute_df[close_col].shift(1) >= alert_day_minute_df[vwap_col].shift(1))
            ]

            if not triggered_minutes.empty:
                first_alert_minute = triggered_minutes.iloc[0]
                alert_time = first_alert_minute.name
                alert_price = first_alert_minute[close_col]
                
                playbook_name = get_val(upthrust_rejection_params.get('alert_playbook_name'), 'EXIT_INTRADAY_UPTHRUST_REJECTION')
                alert_code = get_val(upthrust_rejection_params.get('alert_code'), 103)
                severity_level = get_val(upthrust_rejection_params.get('severity_level'), 3)

                record = self._create_signal_record(
                    stock_code=stock_code,
                    trade_time=alert_time,
                    timeframe=minute_tf,
                    strategy_name="INTRADAY_RISK_ALERT",
                    close_price=alert_price,
                    entry_signal=False,
                    exit_signal_code=alert_code,
                    exit_severity_level=severity_level,
                    exit_signal_reason=f"盘中跌破VWAP, 由前一日({setup_date.date()})创近期新高触发监控",
                    triggered_playbooks=[playbook_name],
                    context_snapshot={'close': alert_price, 'vwap': first_alert_minute[vwap_col]},
                )

                all_alerts.append(record)
                logger.info(f"         - [风险警报!] 日期: {monitoring_date} | 时间: {alert_time.time()} | 价格: {alert_price:.2f} | 规则: {playbook_name}")
                continue 

        return all_alerts

    def _merge_and_deduplicate_signals(self, daily_records: List[Dict], intraday_records: List[Dict]) -> List[Dict]:
        if not daily_records and not intraday_records:
            return daily_records or intraday_records
        signals_by_day = defaultdict(dict)
        def get_trade_date(trade_time_value: Any) -> Optional[datetime.date]:
            try:
                if isinstance(trade_time_value, str):
                    return pd.to_datetime(trade_time_value).date()
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
        【V117.33 数据流修复版】
        - 核心修正: 确保战术引擎的输出(daily_analysis_df)与输入的日线数据(df_daily_prepared)
                    拥有完全相同的时间索引，即使最新几天没有计算出任何分数。
        - 新逻辑:
          1. 在调用战术引擎后，获取其分析结果 daily_analysis_df。
          2. 使用原始日线数据的索引，对 daily_analysis_df 进行 .reindex() 操作。
          3. 这会强制为缺失的日期（如7月14日）补上空行，确保 self.daily_analysis_df
             的时间范围与分钟线数据保持一致。
        - 收益: 解决了因日线分析结果只到7月11日，导致后续分钟线引擎无法找到当天“预备日”的根本问题。
        """
        # 步骤1: 直接使用已经融合了战略信号的日线数据
        df_daily_prepared = all_dfs.get('D')
        if df_daily_prepared is None or df_daily_prepared.empty:
            return []

        print(f"--- [引擎2-调试] 战术引擎接收到的日线数据时间范围到: {df_daily_prepared.index.max()}")

        # 步骤2: 使用完全准备好的日线数据调用战术引擎，生成每日分析结果
        daily_analysis_df, atomic_signals = self.tactical_engine.apply_strategy(df_daily_prepared, self.tactical_config)
        
        if daily_analysis_df is None or daily_analysis_df.empty:
            print("--- [引擎2-调试] 战术引擎返回了空的分析结果。")
            self.daily_analysis_df = pd.DataFrame(index=df_daily_prepared.index) # 创建一个带索引的空df
            return []
        
        print(f"--- [引擎2-调试] 战术引擎原始分析结果时间范围到: {daily_analysis_df.index.max()}")

        # ▼▼▼【代码修改 V117.33】: 强制对齐日线分析结果的索引 ▼▼▼
        # 使用原始日线数据的索引来“校准”分析结果的索引。
        # 这能确保即使最新几天没有生成分数，self.daily_analysis_df 中也存在这些日期行（值为NaN）。
        # 这是让后续分钟线引擎能够找到当天“预备日”的关键。
        self.daily_analysis_df = daily_analysis_df.reindex(df_daily_prepared.index)
        # 对于 reindex 后产生的 NaN，用 0 填充 entry_score，用 False 填充布尔列，以避免后续操作出错
        if 'entry_score' in self.daily_analysis_df.columns:
            self.daily_analysis_df['entry_score'].fillna(0, inplace=True)
        bool_cols = self.daily_analysis_df.select_dtypes(include='bool').columns
        for col in bool_cols:
            self.daily_analysis_df[col].fillna(False, inplace=True)
        
        print(f"--- [引擎2-调试] reindex后，最终 self.daily_analysis_df 时间范围到: {self.daily_analysis_df.index.max()}")
        # ▲▲▲【代码修改 V117.33】▲▲▲

        # 步骤4: 调用波段跟踪模拟器，生成包含交易动作的最终DataFrame
        # 注意：这里传递原始的、没有补全NaN的 daily_analysis_df，因为模拟器可能不需要未来的空行
        df_with_tracking = self.tactical_engine.simulate_wave_tracking(daily_analysis_df, self.tactical_config)
        
        # 步骤5: 使用带有交易动作的DataFrame来准备数据库记录
        return self.tactical_engine.prepare_db_records(stock_code, df_with_tracking, atomic_signals, params=self.tactical_config, result_timeframe='D')

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
        resonance_params = self.tactical_config.get('strategy_params', {}).get('trend_follow', {}).get('multi_level_resonance_params', {})
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
        daily_score_threshold = self.tactical_config.get('entry_scoring_params', {}).get('score_threshold', 100)
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
        tp_params = self.tactical_config.get('strategy_params', {}).get('trend_follow', {}).get('intraday_take_profit_params', {})
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
        resonance_config = self.tactical_config.get('strategy_params', {}).get('trend_follow', {}).get('multi_level_resonance_params', {})
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
        【V117.25 逻辑统一版】
        - 核心修复: 废弃了特殊的 _merge_and_deduplicate_signals 合并逻辑，
                    改为采用与主方法 run_for_stock 完全相同的、简单的信号合并方式。
        - 解决方案: 直接复制 run_for_stock 中的信号合并代码，确保调试方法
                    100% 模拟生产环境的信号生成与合并流程。
        - 收益: 保证了调试结果与生产环境的绝对一致性，让调试真正变得可靠。
        """
        print("=" * 80)
        print(f"--- [历史回溯调试启动 (V117.25 逻辑统一版)] ---")
        print(f"  - 股票代码: {stock_code}")
        print(f"  - 目标时段: {start_date} to {end_date}")
        print("=" * 80)

        try:
            # 步骤 1: 获取全量历史数据 (逻辑保持不变)
            print(f"\n[步骤 1/3] 正在准备从最早到 {end_date} 的所有时间周期数据...")
            all_dfs = await self.indicator_service._prepare_base_data_and_indicators(
                stock_code, self.merged_config, trade_time=end_date
            )
            if 'D' not in all_dfs or all_dfs['D'].empty:
                print(f"[错误] 无法获取 {stock_code} 的日线数据，调试终止。")
                return
            print("[成功] 所有原始数据和指标准备就绪。")

            # 步骤 2: 完整运行所有五个引擎 (逻辑保持不变)
            print("\n[步骤 2/3] 正在完整运行所有五个策略引擎...")
            
            print("  - 引擎1 (周线战略) 启动...")
            strategic_signals_df = self._run_strategic_engine(all_dfs.get('W'))
            print(f"  - 引擎1 (周线战略) 运行完毕，生成 {len(strategic_signals_df)} 条周线分析记录。")

            print("  - [数据流转] 开始将周线战略信号注入日线数据...")
            all_dfs['D'] = self._merge_strategic_signals_to_daily(all_dfs['D'], strategic_signals_df)
            print("  - [数据流转] 完成。")

            print("  - 引擎2 (日线战术) 启动...")
            tactical_records = self._run_tactical_engine(stock_code, all_dfs)
            print(f"  - 引擎2 (日线战术) 运行完毕，生成 {len(tactical_records)} 条日线信号记录。")

            print("  - 引擎3 (分钟共振买入) 启动...")
            resonance_entry_records = self._run_intraday_resonance_engine(stock_code, all_dfs)
            print(f"  - 引擎3 (分钟共振买入) 运行完毕，生成 {len(resonance_entry_records)} 条记录。")

            print("  - 引擎4 (分钟风险预警) 启动...")
            risk_alert_records = self._run_intraday_alert_engine(stock_code, all_dfs)
            print(f"  - 引擎4 (分钟风险预警) 运行完毕，生成 {len(risk_alert_records)} 条记录。")

            print("  - 引擎5 (分钟买入确认) 启动...")
            confirmation_entry_records = self._run_intraday_entry_engine(stock_code, all_dfs)
            print(f"  - 引擎5 (分钟买入确认) 运行完毕，生成 {len(confirmation_entry_records)} 条记录。")

            # ▼▼▼【代码修改 V117.25】: 使用与 run_for_stock 完全相同的合并逻辑 ▼▼▼
            # 将所有买入信号（日线、分钟共振、分钟确认）简单合并
            all_entry_records = tactical_records + resonance_entry_records + confirmation_entry_records
            
            # 将所有买入信号与所有风险信号合并
            all_records = all_entry_records + risk_alert_records
            # ▲▲▲【代码修改 V117.25】▲▲▲
            
            print(f"[成功] 所有引擎运行完毕，共生成 {len(all_records)} 条原始信号记录。")

            # 步骤 3: 筛选并展示目标时段的信号 (逻辑保持不变)
            print(f"\n[步骤 3/3] 正在筛选并展示目标时段 ({start_date} to {end_date}) 的所有信号...")
            
            if not all_records:
                print("[信息] 引擎未生成任何信号记录。")
                return

            start_dt = pd.to_datetime(start_date, utc=True)
            end_dt = pd.to_datetime(end_date, utc=True).replace(hour=23, minute=59, second=59)
            
            debug_period_records = []
            for rec in all_records:
                # 强制将 trade_time 转为带时区的 datetime 对象，以进行安全比较
                rec_time = pd.to_datetime(rec['trade_time'])
                if not rec_time.tzinfo:
                    rec_time = rec_time.tz_localize('UTC')
                else:
                    rec_time = rec_time.tz_convert('UTC')

                if start_dt <= rec_time <= end_dt:
                    debug_period_records.append(rec)

            if not debug_period_records:
                print(f"[信息] 在指定时段 {start_date} to {end_date} 内没有找到任何信号。")
                return
            
            debug_period_records.sort(key=lambda x: pd.to_datetime(x['trade_time']))

            print("\n" + "="*30 + " [全流程信号透视报告] " + "="*30)
            for record in debug_period_records:
                time_obj = pd.to_datetime(record['trade_time'])
                time_str = time_obj.strftime('%Y-%m-%d %H:%M:%S %Z')
                tf = record['timeframe']
                
                if record.get('entry_signal'):
                    score = record.get('entry_score', 0.0)
                    playbooks = record.get('triggered_playbooks', [])
                    signal_type = "买入信号"
                    details = f"得分: {score:<7.2f} | 剧本: {', '.join(playbooks)}"
                    print(f"{time_str} [周期:{tf:>3s}] [类型:{signal_type:<6s}] | {details}")
                
                elif record.get('exit_signal_code', 0) > 0:
                    severity = record.get('exit_severity_level', 0)
                    reason = record.get('exit_signal_reason', 'N/A')
                    signal_type = f"卖出警报(L{severity})"
                    details = f"原因: {reason}"
                    print(f"{time_str} [周期:{tf:>3s}] [类型:{signal_type:<6s}] | {details}")

            print(f"--- [历史回溯调试完成] ---")
            print("=" * 80)

        except Exception as e:
            print(f"[严重错误] 在执行历史回溯调试时发生异常: {e}")
            import traceback
            traceback.print_exc()



