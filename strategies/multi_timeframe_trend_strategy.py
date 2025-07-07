# 文件: strategies/multi_timeframe_trend_strategy.py
# 版本: V6.7 - 分级止盈系统

import asyncio
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
        # ... (构造函数 __init__ 保持不变) ...
        tactical_config_path = 'config/trend_follow_strategy.json'
        strategic_config_path = 'config/weekly_trend_follow_strategy.json'
        self.tactical_config = load_strategy_config(tactical_config_path)
        self.strategic_config = load_strategy_config(strategic_config_path)
        base_merged_fe_params = self._merge_feature_engineering_configs(
            self.tactical_config.get('feature_engineering_params', {}),
            self.strategic_config.get('feature_engineering_params', {})
        )
        resonance_indicators = self._discover_resonance_indicators(self.tactical_config)
        take_profit_indicators = self._discover_take_profit_indicators(self.tactical_config)
        temp_indicators = self._merge_indicators(base_merged_fe_params.get('indicators', {}), resonance_indicators)
        final_indicators = self._merge_indicators(temp_indicators, take_profit_indicators)
        base_merged_fe_params['indicators'] = final_indicators
        self.merged_config = deepcopy(self.tactical_config)
        self.merged_config['feature_engineering_params'] = base_merged_fe_params
        if 'strategy_playbooks' in self.strategic_config:
            self.merged_config['strategy_playbooks'] = deepcopy(self.strategic_config['strategy_playbooks'])
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
        logger.info(f"--- 开始为【{stock_code}】执行三级引擎分析 (V6.7) ---")
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
        logger.info(f"\n--- 引擎4: 开始运行【执行引擎-止盈】(分钟线)... ---")
        take_profit_records = self._run_intraday_take_profit_engine(stock_code, all_dfs)
        logger.info(f"--- 引擎4: 【执行引擎-止盈】运行完毕，生成 {len(take_profit_records)} 条分钟线止盈信号。 ---")
        logger.info(f"\n--- 信号整合: 开始合并日线与分钟线信号...")
        final_entry_records = self._merge_and_deduplicate_signals(tactical_records, execution_records)
        all_records = final_entry_records + take_profit_records
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
        if strategic_signals_df is None or strategic_signals_df.empty: return df_daily
        df_daily_copy = df_daily.copy()
        df_merged = pd.merge_asof(left=df_daily_copy.sort_index(), right=strategic_signals_df.sort_index(), left_index=True, right_index=True, direction='backward')
        for col in strategic_signals_df.columns:
            if col not in df_merged.columns: continue
            if col.startswith(('playbook_', 'signal_', 'state_', 'event_', 'filter_')):
                if col == 'signal_breakout_trigger_W':
                    new_col_name = 'BASE_SIGNAL_BREAKOUT_TRIGGER'
                    df_merged.rename(columns={col: new_col_name}, inplace=True)
                    df_merged[new_col_name] = df_merged[new_col_name].fillna(False).astype(bool)
                else: df_merged[col] = df_merged[col].fillna(False).astype(bool)
            elif col.startswith(('washout_score_', 'rejection_signal_')):
                df_merged[col] = df_merged[col].fillna(0).astype(int)
        return df_merged
    def _run_tactical_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        final_df, atomic_signals = self.tactical_engine.apply_strategy(all_dfs, self.tactical_config)
        self.daily_analysis_df = final_df
        if final_df is None or final_df.empty: return []
        return self.tactical_engine.prepare_db_records(stock_code, final_df, atomic_signals, params=self.tactical_config, result_timeframe='D')
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
