# 文件: strategies/multi_timeframe_trend_strategy.py
# 版本: V6.7 - 分级止盈系统

import io       # 导入 io
import sys      # 导入 sys
import re       # 导入 re
from contextlib import redirect_stdout # 导入 redirect_stdout
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import json
import logging
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import numpy as np
from scipy.signal import find_peaks

from services.indicator_services import IndicatorService
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

    # ▼▼▼【代码修改】: 报告生成函数重大升级，以支持分级止盈 ▼▼▼
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
        【V104 终局 · 执行版】
        - 核心升级: 将旧的“执行引擎-止盈”替换为全新的“执行引擎-风险预警”，实现对特定结构风险的盘中实时狙击。
        """
        logger.info(f"--- 开始为【{stock_code}】执行四级引擎分析 (V104) ---")
        logger.info(f"--- 准备阶段: 调用 IndicatorService 统一准备所有数据... ---")
        all_dfs = await self.indicator_service._prepare_base_data_and_indicators(stock_code, self.merged_config, trade_time)
        if 'D' not in all_dfs or 'W' not in all_dfs:
            logger.warning(f"[{stock_code}] 核心数据(周线或日线)准备失败，分析终止。")
            return None
        logger.info(f"\n--- 引擎1: 开始运行【战略引擎】(周线)... ---")
        strategic_signals_df = self._run_strategic_engine(all_dfs['W'])
        logger.info(f"--- 引擎1: 【战略引擎】运行完毕。---")
        logger.info(f"\n--- 数据流转: 整合战略信号到日线数据... ---")
        all_dfs['D'] = self._merge_strategic_signals_to_daily(all_dfs['D'], strategic_signals_df)
        logger.info(f"\n--- 引擎2: 开始运行【战术引擎】(日线)... ---")
        tactical_records = self._run_tactical_engine(stock_code, all_dfs)
        logger.info(f"--- 引擎2: 【战术引擎】运行完毕，生成 {len(tactical_records)} 条日线买入信号。 ---")
        logger.info(f"\n--- 引擎3: 开始运行【执行引擎-买入】(分钟线)... ---")
        execution_records = self._run_intraday_resonance_engine(stock_code, all_dfs)
        logger.info(f"--- 引擎3: 【执行引擎-买入】运行完毕，生成 {len(execution_records)} 条分钟线买入信号。 ---")
        
        # ▼▼▼【代码修改 V104】: 调用全新的分钟级风险预警引擎 ▼▼▼
        logger.info(f"\n--- 引擎4: 开始运行【通用盘中预警引擎】(分钟线)... ---")
        risk_alert_records = self._run_intraday_alert_engine(stock_code, all_dfs) # 调用新函数
        logger.info(f"--- 引擎4: 【通用盘中预警引擎】运行完毕，生成 {len(risk_alert_records)} 条分钟线风险警报。 ---")

        logger.info(f"\n--- 信号整合: 开始合并日线与分钟线信号...")
        final_entry_records = self._merge_and_deduplicate_signals(tactical_records, execution_records)
        
        # ▼▼▼【代码修改 V104】: 将风险警报合并到最终记录中 ▼▼▼
        all_records = final_entry_records + risk_alert_records

        if all_records:
            latest_trade_date = max(pd.to_datetime(rec['trade_time']).date() for rec in all_records)
            latest_records = [
                record for record in all_records
                if pd.to_datetime(record['trade_time']).date() == latest_trade_date
            ]
            if latest_records:
                logger.info(f"\n--- 报告生成: 为最新交易日 {latest_trade_date} 的 {len(latest_records)} 条信号生成分析报告...")
                print(f"--- 分析报告仅展示最新交易日({latest_trade_date})的信号 ---")
                for record in latest_records:
                    report_text = self._generate_analysis_report(record)
                    record['analysis_text'] = report_text
                    print("----------------------------------------------------")
                    print(report_text)
                    print("----------------------------------------------------")
        logger.info(f"\n--- 【{stock_code}】所有引擎分析完成，共生成 {len(all_records)} 条最终信号记录。 ---")
        return all_records if all_records else None

    async def run_for_stock(self, stock_code: str, trade_time: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        【V106.1 终局 · 增强版】
        - 核心修正: 明确了日线和分钟线买入信号的关系。
        - 日线信号: 作为基础战略决策，必须被记录。
        - 分钟线信号: 作为“增强信号”，在日线高分的基础上寻找盘中确认点。
                        它的触发会生成一个更高分的信号，在信号合并时优先采纳，但不会“否定”未被确认的日线信号。
        """
        logger.info(f"--- 开始为【{stock_code}】执行五级引擎分析 (V106.1) ---")
        
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
        
        # --- 数据流转 ---
        logger.info(f"\n--- 数据流转: 整合战略信号到日线数据... ---")
        all_dfs['D'] = self._merge_strategic_signals_to_daily(all_dfs['D'], strategic_signals_df)
        
        # --- 引擎2: 战术引擎 (日线) ---
        logger.info(f"\n--- 引擎2: 开始运行【战术引擎】(日线)... ---")
        # ▼▼▼【代码修改 V106.1】: 明确 tactical_records 是必须保留的日线信号 ▼▼▼
        tactical_records = self._run_tactical_engine(stock_code, all_dfs)
        logger.info(f"--- 引擎2: 【战术引擎】运行完毕，生成 {len(tactical_records)} 条日线买入信号。 ---")
        
        # --- 引擎3: 执行引擎-买入 (分钟线共振) ---
        logger.info(f"\n--- 引擎3: 开始运行【执行引擎-买入】(分钟线)... ---")
        resonance_entry_records = self._run_intraday_resonance_engine(stock_code, all_dfs)
        logger.info(f"--- 引擎3: 【执行引擎-买入】运行完毕，生成 {len(resonance_entry_records)} 条分钟线买入信号。 ---")
        
        # --- 引擎4: 通用盘中预警引擎 (分钟线风险) ---
        logger.info(f"\n--- 引擎4: 开始运行【通用盘中预警引擎】(分钟线)... ---")
        risk_alert_records = self._run_intraday_alert_engine(stock_code, all_dfs)
        logger.info(f"--- 引擎4: 【通用盘中预警引擎】运行完毕，生成 {len(risk_alert_records)} 条分钟线风险警报。 ---")

        # --- 引擎5: 通用盘中买入确认引擎 (分钟线增强) ---
        logger.info(f"\n--- 引擎5: 开始运行【通用盘中买入确认引擎】(分钟线)... ---")
        confirmation_entry_records = self._run_intraday_entry_engine(stock_code, all_dfs)
        logger.info(f"--- 引擎5: 【通用盘中买入确认引擎】运行完毕，生成 {len(confirmation_entry_records)} 条分钟线买入确认信号。 ---")

        # --- 信号整合 ---
        logger.info(f"\n--- 信号整合: 开始合并日线与分钟线信号...")
        # ▼▼▼【代码修改 V106.1】: 整合所有分钟级别的买入信号 ▼▼▼
        all_intraday_entry_records = resonance_entry_records + confirmation_entry_records
        
        # ▼▼▼【代码修改 V106.1】: 使用合并函数，它会智能地用分钟线信号覆盖同一天的日线信号 ▼▼▼
        # 这里的逻辑是：
        # 1. tactical_records 包含了所有日线买入信号。
        # 2. all_intraday_entry_records 包含了所有分钟线买入信号。
        # 3. _merge_and_deduplicate_signals 会优先保留分钟线信号。
        # 4. 如果某一天只有日线信号，它会被保留。
        # 5. 如果某一天既有日线信号又有分钟线信号，只有分钟线信号会被保留。
        # 这完美实现了“增强”而非“替代”的逻辑。
        final_entry_records = self._merge_and_deduplicate_signals(tactical_records, all_intraday_entry_records)
        
        # 将最终的买入信号和风险信号合并
        all_records = final_entry_records + risk_alert_records

        # --- 报告生成 ---
        if all_records:
            latest_trade_date = max(pd.to_datetime(rec['trade_time']).date() for rec in all_records)
            latest_records = [
                record for record in all_records
                if pd.to_datetime(record['trade_time']).date() == latest_trade_date
            ]
            if latest_records:
                logger.info(f"\n--- 报告生成: 为最新交易日 {latest_trade_date} 的 {len(latest_records)} 条信号生成分析报告...")
                print(f"--- 分析报告仅展示最新交易日({latest_trade_date})的信号 ---")
                for record in latest_records:
                    report_text = self._generate_analysis_report(record)
                    record['analysis_text'] = report_text
                    print("----------------------------------------------------")
                    print(report_text)
                    print("----------------------------------------------------")
                    
        logger.info(f"\n--- 【{stock_code}】所有引擎分析完成，共生成 {len(all_records)} 条最终信号记录。 ---")
        return all_records if all_records else None

    def _run_intraday_entry_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        【V108 终局 · 确认版】
        - 核心升级: 在VWAP突破的基础上，增加了“成交量确认”和“波动率突破”两个辅助条件，
                    形成“铁三角”共振，大幅提升分钟线买入信号的质量。
        """
        all_confirmations = []
        # --- 步骤0: 加载配置和数据 ---
        entry_params = self.tactical_engine._get_params_block(self.tactical_config, 'intraday_entry_params', {})
        get_val = self.tactical_engine._get_param_value
        
        if not get_val(entry_params.get('enabled'), False):
            return []
        if self.daily_analysis_df is None or self.daily_analysis_df.empty:
            return []

        # --- 步骤1: 识别所有日线高分的“预备日” ---
        daily_score_threshold = get_val(entry_params.get('daily_score_threshold'), 100)
        setup_days_df = self.daily_analysis_df[self.daily_analysis_df['entry_score'] >= daily_score_threshold]
        if setup_days_df.empty:
            return []
            
        logger.info(f"    - [买入确认任务] 发现 {len(setup_days_df)} 个日线高分预备日，启动当日盘中监控...")

        # --- 步骤2: 遍历每个预备日，监控当天分钟行情 ---
        minute_tf = str(get_val(entry_params.get('timeframe'), '5'))
        minute_df = all_dfs.get(minute_tf)
        if minute_df is None or minute_df.empty:
            return []

        rules = entry_params.get('confirmation_rules', {})
        min_time_after_open = get_val(rules.get('min_time_after_open'), 15)

        for setup_date, setup_row in setup_days_df.iterrows():
            alert_day_minute_df = minute_df[minute_df.index.date == setup_date.date()].copy()
            if alert_day_minute_df.empty: continue

            # --- 步骤3: 构建三位一体的分钟线确认逻辑 ---
            final_confirmation_signal = pd.Series(True, index=alert_day_minute_df.index)

            close_col = f'close_{minute_tf}'
            volume_col = f'volume_{minute_tf}'

            # 条件1: VWAP突破
            vwap_rule = rules.get('vwap_reclaim', {})
            if get_val(vwap_rule.get('enabled'), False):
                vwap_col = f'VWAP_{minute_tf}'
                if vwap_col in alert_day_minute_df.columns and close_col in alert_day_minute_df.columns:
                    final_confirmation_signal &= (alert_day_minute_df[close_col] > alert_day_minute_df[vwap_col])
                else:
                    final_confirmation_signal &= False

            # 条件2: 成交量确认
            vol_rule = rules.get('volume_confirmation', {})
            if get_val(vol_rule.get('enabled'), False):
                vol_ma_period = get_val(vol_rule.get('ma_period'), 21)
                vol_ma_col = f'VOL_MA_{vol_ma_period}_{minute_tf}'
                if vol_ma_col in alert_day_minute_df.columns and volume_col in alert_day_minute_df.columns:
                    final_confirmation_signal &= (alert_day_minute_df[volume_col] > alert_day_minute_df[vol_ma_col])

            # 条件3: 波动率突破
            vola_rule = rules.get('volatility_breakout', {})
            if get_val(vola_rule.get('enabled'), False):
                bbw_col = f'BBW_21_2.0_{minute_tf}'
                if bbw_col in alert_day_minute_df.columns:
                    # ... 波动率计算逻辑保持不变，因为它内部已经使用了bbw_col ...
                    squeeze_threshold = alert_day_minute_df[bbw_col].rolling(...).quantile(...)
                    is_expanding_from_squeeze = (alert_day_minute_df[bbw_col] > alert_day_minute_df[bbw_col].shift(1)) & \
                                                (alert_day_minute_df[bbw_col].shift(1) < squeeze_threshold)
                    final_confirmation_signal &= is_expanding_from_squeeze

            # 过滤掉开盘初期的噪音，并寻找首次触发点
            market_open_time = setup_date.replace(hour=9, minute=30, second=0)
            monitoring_start_time = market_open_time + pd.Timedelta(minutes=min_time_after_open)
            
            # 寻找首次从False变为True的那个点
            triggered_minutes = alert_day_minute_df[
                (alert_day_minute_df.index >= monitoring_start_time) &
                (final_confirmation_signal == True) & 
                (final_confirmation_signal.shift(1) == False)
            ]

            if not triggered_minutes.empty:
                first_confirmation_minute = triggered_minutes.iloc[0]
                confirm_time = first_confirmation_minute.name
                confirm_price = first_confirmation_minute[close_col]
                
                # --- 步骤4: 生成增强的买入信号记录 ---
                daily_score = setup_row.get('entry_score', 0)
                bonus_score = get_val(entry_params.get('bonus_score'), 50)
                final_score = daily_score + bonus_score
                
                daily_playbooks = [p.replace('playbook_', '') for p in setup_row.index if p.startswith('playbook_') and setup_row[p] is True]
                
                record = {
                    "stock_code": stock_code,
                    "trade_time": confirm_time.to_pydatetime(),
                    "timeframe": minute_tf,
                    "strategy_name": get_val(entry_params.get('signal_name'), 'INTRADAY_ENTRY_CONFIRMATION'),
                    "close_price": sanitize_for_json(confirm_price),
                    "entry_score": final_score,
                    "entry_signal": True,
                    "exit_signal_code": 0,
                    "exit_severity_level": 0,
                    "exit_signal_reason": None,
                    "triggered_playbooks": list(set(daily_playbooks + [get_val(entry_params.get('playbook_name'), 'ENTRY_INTRADAY_CONFIRMATION')])),
                    "context_snapshot": sanitize_for_json({'close': confirm_price, 'daily_score': daily_score, 'bonus': bonus_score}),
                }
                all_confirmations.append(record)
                logger.info(f"         - [买入确认!] 日期: {setup_date.date()} | 时间: {confirm_time.time()} | 价格: {confirm_price:.2f} | 最终得分: {final_score:.0f}")
                continue 

        return all_confirmations

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
        【V85.0 波段跟踪集成版】
        - 核心升级: 在战术引擎生成每日信号后，立即调用波段跟踪模拟器，将无状态信号转化为有状态的交易动作。
        """
        # 步骤1: 融合周线战略信号到日线数据
        strategic_signals_df = self._run_strategic_engine(all_dfs.get('W'))
        df_daily_prepared = self._merge_strategic_signals_to_daily(all_dfs['D'], strategic_signals_df)

        # 步骤2: 融合分钟线信号（如VWAP支撑）到日线数据
        df_daily_prepared = self._prepare_intraday_signals(all_dfs, self.tactical_config)

        # 步骤3: 使用完全准备好的日线数据调用战术引擎，生成每日分析结果
        daily_analysis_df, atomic_signals = self.tactical_engine.apply_strategy(df_daily_prepared, self.tactical_config)
        
        self.daily_analysis_df = daily_analysis_df # 缓存每日分析结果供其他引擎使用
        if daily_analysis_df is None or daily_analysis_df.empty: return []

        # ▼▼▼ 注入波段跟踪模拟器 ▼▼▼
        # 步骤4: 调用波段跟踪模拟器，生成包含交易动作的最终DataFrame
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
            record = self._prepare_intraday_db_record(stock_code, timestamp, row, resonance_params)
            record['triggered_playbooks'] = combined_playbooks
            daily_score = sanitize_for_json(row.get('daily_entry_score', 0.0))
            resonance_score = sanitize_for_json(resonance_params.get('score', 0.0))
            total_score = daily_score + resonance_score
            record['entry_score'] = total_score
            db_records.append(record)
        return db_records

    # ▼▼▼【代码修改】: 止盈引擎重构，实现三级警报系统 ▼▼▼
    def _run_intraday_take_profit_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        tp_params = self.tactical_config.get('strategy_params', {}).get('trend_follow', {}).get('intraday_take_profit_params', {})
        if not tp_params.get('enabled', False): return []
        
        tf = tp_params.get('timeframe')
        if not tf or tf not in all_dfs or all_dfs[tf].empty: return []
        
        df = all_dfs[tf].copy()

        # 1. 数据融合：将日线和高阶分钟线数据融合到当前检查周期
        dynamics_timeframes = ['60', '30']
        for health_tf in dynamics_timeframes:
            if health_tf in all_dfs and not all_dfs[health_tf].empty:
                df_right = all_dfs[health_tf].copy()
                rename_map = {col: f"{col}_{health_tf}" for col in df_right.columns}
                df_right.rename(columns=rename_map, inplace=True)
                df = pd.merge_asof(left=df, right=df_right, left_index=True, right_index=True, direction='backward')
        
        # 融合日线关键支撑位
        daily_support_ma = 'EMA_55_D'
        if 'D' in all_dfs and daily_support_ma in all_dfs['D'].columns:
            df = pd.merge_asof(left=df, right=all_dfs['D'][[daily_support_ma]], left_index=True, right_index=True, direction='backward')

        # 2. 计算趋势动态
        df = self._calculate_trend_dynamics(df, dynamics_timeframes, ema_period=34, slope_window=5)

        # 3. 定义各级警报信号
        signals = []
        
        # 警报等级 1: 趋势减速 (黄色预警)
        is_still_rising = df.get('ema_slope_30', 0) > 0
        is_decelerating = df.get('ema_accel_30', 0) < 0
        was_accelerating = df.get('ema_accel_30', 0).shift(1) >= 0
        level_1_signal = is_still_rising & is_decelerating & was_accelerating
        if level_1_signal.any():
            signals.append({'level': 1, 'reason': '30分钟趋势加速度转负', 'signal': level_1_signal})

        # 警报等级 2: 短期指标转弱 (橙色警报)
        p = [12, 26, 9] # 假设使用15分钟MACD
        macd_col, signal_col = f'MACD_{p[0]}_{p[1]}_{p[2]}', f'MACDs_{p[0]}_{p[1]}_{p[2]}'
        if macd_col in df.columns and signal_col in df.columns:
            base_signal = (df[macd_col] < df[signal_col]) & (df[macd_col].shift(1) >= df[signal_col].shift(1))
            is_trend_deteriorating = df.get('trend_health_30', False) == False
            level_2_signal = base_signal & is_trend_deteriorating
            if level_2_signal.any():
                signals.append({'level': 2, 'reason': f'{tf}分钟MACD死叉且30分钟趋势不健康', 'signal': level_2_signal})

        # 警报等级 3: 跌破日线关键支撑 (红色警报)
        if daily_support_ma in df.columns:
            level_3_signal = (df['close'] < df[daily_support_ma]) & (df['close'].shift(1) >= df[daily_support_ma].shift(1))
            if level_3_signal.any():
                signals.append({'level': 3, 'reason': f'价格跌破日线关键支撑({daily_support_ma})', 'signal': level_3_signal})

        # 4. 合并与去重
        if not signals: return []
        
        df['exit_severity_level'] = 0
        df['exit_signal_reason'] = ''
        
        # 按严重性从高到低应用信号，高级别信号会覆盖低级别信号
        for s in sorted(signals, key=lambda x: x['level'], reverse=True):
            df.loc[s['signal'], 'exit_severity_level'] = s['level']
            df.loc[s['signal'], 'exit_signal_reason'] = s['reason']
            
        triggered_df = df[df['exit_severity_level'] > 0].copy()
        if triggered_df.empty: return []

        # 5. 准备数据库记录
        db_records = []
        for timestamp, row in triggered_df.iterrows():
            record = {
                "stock_code": stock_code,
                "trade_time": timestamp.to_pydatetime(),
                "timeframe": tf,
                "strategy_name": tp_params.get('signal_name', 'INTRADAY_TAKE_PROFIT'),
                "close_price": sanitize_for_json(row.get('close')),
                "entry_score": 0.0,
                "entry_signal": False,
                "exit_signal_code": 100 + int(row.get('exit_severity_level', 0)), # 使用等级作为code的一部分
                "exit_severity_level": sanitize_for_json(row.get('exit_severity_level')),
                "exit_signal_reason": sanitize_for_json(row.get('exit_signal_reason')),
                "triggered_playbooks": [f"EXIT_LEVEL_{int(row.get('exit_severity_level', 0))}"],
                "context_snapshot": sanitize_for_json({'close': row.get('close'), 'reason': row.get('exit_signal_reason')}),
            }
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

    def _prepare_intraday_db_record(self, stock_code: str, timestamp: pd.Timestamp, row: pd.Series, params: dict) -> Dict[str, Any]:
        # ... (此函数保持不变, 但注意在调用它的地方，要为新字段提供默认值) ...
        signal_name = params.get('signal_name', 'UNKNOWN_RESONANCE')
        trigger_tf = params['levels'][-1]['tf']
        native_utc_datetime: datetime = timestamp.to_pydatetime()
        record = {
            "stock_code": stock_code,
            "trade_time": native_utc_datetime,
            "timeframe": trigger_tf,
            "strategy_name": signal_name,
            "close_price": sanitize_for_json(row.get('close')),
            "entry_score": sanitize_for_json(params.get('score', 0.0)),
            "entry_signal": True,
            "exit_signal_code": 0,
            "exit_severity_level": 0, # 为买入信号设置默认值
            "exit_signal_reason": None, # 为买入信号设置默认值
            "triggered_playbooks": [signal_name],
            "context_snapshot": sanitize_for_json({'close': row.get('close')}),
        }
        return record

    async def debug_run_for_period(self, stock_code: str, start_date: str, end_date: str):
        """
        【V85.0 波段跟踪集成版】
        - 核心升级: 在调试模式下同样集成波段跟踪模拟器，并增加专门的日志输出，清晰展示每个交易动作。
        """
        print("=" * 80)
        print(f"--- [历史回溯调试启动 (V85.0 波段跟踪版)] ---")
        print(f"  - 股票代码: {stock_code}")
        print(f"  - 目标时段: {start_date} to {end_date}")
        print("=" * 80)

        try:
            # 步骤 1: 获取全量历史数据
            print(f"\n[步骤 1/4] 正在准备从最早到 {end_date} 的所有时间周期数据...")
            all_dfs = await self.indicator_service._prepare_base_data_and_indicators(
                stock_code, self.merged_config, trade_time=end_date
            )
            if 'D' not in all_dfs or all_dfs['D'].empty:
                print(f"[错误] 无法获取 {stock_code} 的日线数据，调试终止。")
                return
            print("[成功] 所有原始数据准备就绪。")

            # 步骤 2: 运行战略引擎和战术引擎，并捕获日志
            print("\n[步骤 2/4] 正在使用全量数据运行引擎 (日志将被捕获并过滤)...")
            
            strategic_signals_df = self._run_strategic_engine(all_dfs.get('W'))
            df_daily_prepared = self._merge_strategic_signals_to_daily(all_dfs['D'], strategic_signals_df)
            df_daily_prepared = self._prepare_intraday_signals(all_dfs, self.tactical_config)
            
            log_capture_buffer = io.StringIO()
            with redirect_stdout(log_capture_buffer):
                # 运行战术引擎，得到每日分析结果
                daily_analysis_df, _ = self.tactical_engine.apply_strategy(df_daily_prepared, self.tactical_config)
                
                # ▼▼▼【代码修改 V85.0】: 在调试模式下同样注入波段跟踪模拟器 ▼▼▼
                if daily_analysis_df is not None and not daily_analysis_df.empty:
                    df_with_tracking = self.tactical_engine.simulate_wave_tracking(daily_analysis_df, self.tactical_config)
                else:
                    df_with_tracking = daily_analysis_df
                # ▲▲▲【代码修改 V85.0】▲▲▲

            captured_logs = log_capture_buffer.getvalue()

            if df_with_tracking is None or df_with_tracking.empty:
                print("[信息] 引擎运行完成，但未生成任何分析结果。")
                print("\n--- [捕获的底层引擎日志] ---\n" + captured_logs)
                return
            print("[成功] 战术引擎及波段跟踪模拟完成。")

            # 步骤 3: 筛选目标时段的分析结果
            print(f"\n[步骤 3/4] 正在筛选目标时段 ({start_date} to {end_date}) 的分析结果...")
            debug_period_df = df_with_tracking.loc[start_date:end_date].copy()
            if debug_period_df.empty:
                print(f"[信息] 在指定时段 {start_date} to {end_date} 内没有找到数据。")
                return
            print(f"[成功] 筛选出 {len(debug_period_df)} 个交易日的分析数据。")

            # 步骤 4: 打印过滤后的日志和交易动作
            print("\n--- [底层引擎日志 (仅显示目标时段相关)] ---")
            # 生成一个从 start_date 开始的所有年份的正则表达式，以匹配多行日志块
            start_year = pd.to_datetime(start_date).year
            current_year = pd.to_datetime(end_date).year
            years_to_match = [str(y) for y in range(start_year, current_year + 2)] # 加到下一年以防跨年
            
            # 匹配如 "====== 日期: 2024-08-01" 或 "--- [V反剧本评估] 详细调试 for 2024-08-01 ---"
            date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})')
            
            # 标记是否应该开始打印
            printing_started = False
            for line in captured_logs.splitlines():
                # 检查日志行是否包含目标年份范围内的日期
                match = date_pattern.search(line)
                if match:
                    log_date_str = match.group(1)
                    try:
                        log_date = pd.to_datetime(log_date_str)
                        if log_date >= pd.to_datetime(start_date):
                            printing_started = True
                    except ValueError:
                        pass # 如果日期格式不正确，则忽略

                # 如果日志行本身不包含日期，但它属于一个从目标日期开始的日志块，也打印它
                # 我们通过检查日志块的开头来判断，比如 "--- [" 或 "    ["
                if printing_started or line.strip().startswith(('---', '    [', '  - ', '======')):
                    print(line)
            print("--- [底层引擎日志结束] ---")

            # ▼▼▼ 新增波段交易动作的调试输出 ▼▼▼
            print("\n" + "="*30 + " [波段跟踪交易动作] " + "="*30)
            trade_actions_in_period = debug_period_df[debug_period_df['trade_action'] != '']
            if trade_actions_in_period.empty:
                print("在指定时段内无交易动作发生。")
            else:
                for timestamp, row in trade_actions_in_period.iterrows():
                    action_str = f"[{row.trade_action}]"
                    price_str = f"价格: {row.close_D:.2f}"
                    pos_str = f"仓位: {row.position_status*100:.0f}%"
                    score_str = f"入场分: {row.entry_score:.0f}" if row.trade_action == 'ENTRY' else ""
                    print(f"{timestamp.strftime('%Y-%m-%d')}: {action_str:<15} {price_str:<18} {pos_str:<12} {score_str}")
            print("=" * 80)

            print(f"--- [历史回溯调试完成] ---")
            print("=" * 80)

        except Exception as e:
            print(f"[严重错误] 在执行历史回溯调试时发生异常: {e}")
            import traceback
            traceback.print_exc()

