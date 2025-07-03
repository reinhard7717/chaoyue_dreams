# 文件: strategies/multi_timeframe_trend_strategy.py
# 版本: V6.4 - 剧本合并修复版

import asyncio
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import json
import logging
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from scipy.signal import find_peaks # 导入find_peaks用于背离检测

from services.indicator_services import IndicatorService
from strategies.trend_following_strategy import TrendFollowStrategy
from strategies.weekly_trend_follow_strategy import WeeklyTrendFollowStrategy
from utils.config_loader import load_strategy_config
from utils.data_sanitizer import sanitize_for_json

logger = logging.getLogger(__name__)

class MultiTimeframeTrendStrategy:

    def __init__(self):
        """
        【V6.4 剧本合并修复版】
        - 核心修复: 修复了分钟线信号未合并日线/周线剧本的问题。
        - 优化: 调整了剧本合并逻辑的位置，使其在分钟线信号生成时就完成，简化了最终的信号合并函数。
        """
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

    def _discover_take_profit_indicators(self, config: Dict) -> Dict:
        """从止盈配置中发现需要计算的指标。"""
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
                indicator_name, params = 'rsi', {'apply_on': [tf], 'periods': [rule['period']]} # 修正为 period
            
            if indicator_name and params and params not in discovered[indicator_name]['configs']:
                discovered[indicator_name]['configs'].append(params)
        
        return json.loads(json.dumps(discovered))
    
    def _discover_resonance_indicators(self, config: Dict) -> Dict:
        # ... 此方法保持不变 ...
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
        # ... 此方法保持不变 ...
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
        # ... 此方法保持不变 ...
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
        # ... 此方法保持不变 ...
        logger.info(f"--- 开始为【{stock_code}】执行三级引擎分析 (V6.4) ---")
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
        
        logger.info(f"\n--- 【{stock_code}】所有引擎分析完成，共生成 {len(all_records)} 条最终信号记录。 ---")
        return all_records if all_records else None

    # ▼▼▼【代码修改】: 简化此方法，因为剧本合并逻辑已前移 ▼▼▼
    def _merge_and_deduplicate_signals(self, daily_records: List[Dict], intraday_records: List[Dict]) -> List[Dict]:
        """
        【V6.4 剧本合并修复版】
        合并日线和分钟线买入信号。
        规则：如果同一天同时存在日线和分钟线信号，优先保留分钟线信号。
        因为分钟线信号记录在生成时已经合并了日线剧本，所以这里的逻辑大大简化。
        """
        if not daily_records and not intraday_records:
            return daily_records or intraday_records

        signals_by_day = defaultdict(dict)

        def get_trade_date(trade_time_value: Any) -> Optional[datetime.date]:
            """辅助函数，无论输入是字符串、datetime对象还是Timestamp，都安全地返回date对象。"""
            try:
                if isinstance(trade_time_value, str):
                    return pd.to_datetime(trade_time_value).date()
                elif hasattr(trade_time_value, 'date'):
                    return trade_time_value.date()
                else:
                    # print(f"  - [信号整合警告] 未知的trade_time类型: {type(trade_time_value)}")
                    return None
            except Exception as e:
                # print(f"  - [信号整合警告] 解析时间失败: {trade_time_value}, 错误: {e}")
                return None

        # 处理日线信号
        for record in daily_records:
            if record.get('entry_signal'):
                trade_date = get_trade_date(record.get('trade_time'))
                if trade_date:
                    signals_by_day[trade_date]['D'] = record
        
        # 处理分钟线信号
        for record in intraday_records:
            if record.get('entry_signal'):
                trade_date = get_trade_date(record.get('trade_time'))
                if trade_date:
                    signals_by_day[trade_date]['M'] = record

        final_records = []
        sorted_dates = sorted(signals_by_day.keys())

        for trade_date in sorted_dates:
            signals = signals_by_day[trade_date]
            
            # ▼▼▼【代码修改】: 简化合并逻辑 ▼▼▼
            if 'M' in signals:
                # 优先保留分钟线信号，因为它已经包含了所有需要的信息（分数、合并后的剧本）
                # print(f"  - [信号整合] 日期 {trade_date}: 发现分钟线信号，优先保留。剧本: {signals['M'].get('triggered_playbooks')}")
                final_records.append(signals['M'])
            elif 'D' in signals:
                # 如果当天只有日线信号，则保留日线信号
                # print(f"  - [信号整合] 日期 {trade_date}: 仅发现日线信号，予以保留。")
                final_records.append(signals['D'])
            # ▲▲▲【代码修改】: 结束 ▲▲▲
        
        return final_records

    def _run_strategic_engine(self, df_weekly: pd.DataFrame) -> pd.DataFrame:
        # ... 此方法保持不变 ...
        if df_weekly is None or df_weekly.empty:
            logger.warning("周线数据为空，战略引擎跳过。")
            return pd.DataFrame()
        return self.strategic_engine.apply_strategy(df_weekly)
    
    def _merge_strategic_signals_to_daily(self, df_daily: pd.DataFrame, strategic_signals_df: pd.DataFrame) -> pd.DataFrame:
        # ... 此方法保持不变 ...
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
        # ... 此方法保持不变 ...
        final_df, atomic_signals = self.tactical_engine.apply_strategy(all_dfs, self.tactical_config)
        self.daily_analysis_df = final_df
        if final_df is None or final_df.empty: return []
        return self.tactical_engine.prepare_db_records(stock_code, final_df, atomic_signals, params=self.tactical_config, result_timeframe='D')

    # ▼▼▼【代码修改】: 此方法为核心修改区域，负责合并剧本 ▼▼▼
    def _run_intraday_resonance_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        # print("\n--- [引擎3-调试 V6.4] 进入 _run_intraday_resonance_engine ---")
        resonance_params = self.tactical_config.get('strategy_params', {}).get('trend_follow', {}).get('multi_level_resonance_params', {})
        if not resonance_params.get('enabled', False):
            # print("    - [引擎3-调试] 结论: 分钟共振剧本在JSON中未启用 (enabled: false)。引擎退出。")
            return []
        if self.daily_analysis_df is None or self.daily_analysis_df.empty:
            # print("    - [引擎3-调试] 结论: 缺少日线策略分析结果 (self.daily_analysis_df)，无法执行分钟线引擎。引擎退出。")
            return []
        levels = resonance_params.get('levels', [])
        if not levels:
            # print("    - [引擎3-调试] 结论: JSON中未定义任何共振层级 (levels为空)。引擎退出。")
            return []
        trigger_tf = levels[-1]['tf']
        if trigger_tf not in all_dfs or all_dfs[trigger_tf].empty:
            # print(f"    - [引擎3-调试] 结论: 缺少或为空的触发周期 '{trigger_tf}' 数据。引擎退出。")
            return []
        df_aligned = all_dfs[trigger_tf].copy()
        for level in levels[:-1]:
            level_tf = level['tf']
            if level_tf in all_dfs and not all_dfs[level_tf].empty:
                df_right = all_dfs[level_tf].copy()
                rename_map = {col: f"{col}_{level_tf}" for col in df_right.columns}
                df_right.rename(columns=rename_map, inplace=True)
                df_aligned = pd.merge_asof(
                    left=df_aligned, right=df_right, left_index=True, right_index=True, direction='backward'
                )
            else:
                # print(f"    - [引擎3-调试] 结论: 共振检查缺少关键的上层周期 '{level_tf}' 数据。引擎退出。")
                return []
        # print("    - [引擎3-调试] 新增步骤: 融合日线趋势、得分及剧本，创建综合过滤器...")
        daily_score_threshold = self.tactical_config.get('entry_scoring_params', {}).get('score_threshold', 100)
        
        # ▼▼▼【代码修改】: 从日线分析结果中，除了分数，还要提取出所有playbook列 ▼▼▼
        daily_playbook_cols = [col for col in self.daily_analysis_df.columns if col.startswith('playbook_')]
        daily_context_cols_to_merge = ['context_mid_term_bullish', 'entry_score'] + daily_playbook_cols
        daily_context_df = self.daily_analysis_df[daily_context_cols_to_merge].copy()
        # ▲▲▲【代码修改】: 结束 ▲▲▲

        is_bullish_trend = daily_context_df['context_mid_term_bullish']
        is_reversal_day = daily_context_df['entry_score'] >= daily_score_threshold
        daily_context_df['is_daily_trend_ok'] = is_bullish_trend | is_reversal_day
        daily_context_df.rename(columns={'entry_score': 'daily_entry_score'}, inplace=True)
        # print(f"      - [引擎3-调试] 日线看涨天数: {is_bullish_trend.sum()}, 日线信号日天数: {is_reversal_day.sum()}, 总合格天数: {daily_context_df['is_daily_trend_ok'].sum()}")
        
        # 将包含分数和剧本的日线上下文合并到分钟线数据中
        df_aligned = pd.merge_asof(
            left=df_aligned,
            right=daily_context_df, # 现在这里包含了分数和所有剧本列
            left_index=True,
            right_index=True,
            direction='backward'
        )
        df_aligned['is_daily_trend_ok'].fillna(False, inplace=True)
        df_aligned['daily_entry_score'].fillna(0, inplace=True)
        for col in daily_playbook_cols: # 对合并过来的剧本列填充NA值
            if col in df_aligned.columns:
                df_aligned[col].fillna(False, inplace=True)

        # print("    - [引擎3-调试] 数据对齐完成。开始逐级检查共振条件...")
        final_signal = pd.Series(True, index=df_aligned.index)
        final_signal &= df_aligned['is_daily_trend_ok']
        # print(f"    - [引擎3-调试] 应用日线综合过滤后，剩余候选信号点: {final_signal.sum()}")
        if final_signal.sum() == 0:
            # print("    - [引擎3-调试] 结论: 所有时间点均不满足日线前提，无需进一步检查分钟线条件。引擎正常结束。")
            return []
        for i, level in enumerate(levels):
            level_tf, level_name = level['tf'], level.get('level_name', f'Level_{i}')
            level_logic, level_conditions = level.get('logic', 'AND').upper(), level.get('conditions', [])
            # print(f"      - [引擎3-调试] 正在检查第 {i+1} 层: '{level_name}' (周期: {level_tf}, 逻辑: {level_logic})")
            level_signal = pd.Series(True if level_logic == 'AND' else False, index=df_aligned.index)
            for cond in level_conditions:
                cond_signal = self._check_single_condition(df_aligned, cond, level_tf)
                if level_logic == 'AND': level_signal &= cond_signal
                else: level_signal |= cond_signal
            final_signal &= level_signal
        # print(f"    - [引擎3-调试] 所有层级检查完毕。最终共振信号触发总次数: {final_signal.sum()}")
        triggered_df = df_aligned[final_signal]
        if triggered_df.empty:
            # print("    - [引擎3-调试] 结论: 没有发现任何满足所有条件的共振信号点。引擎正常结束。")
            return []
        # print(f"    - [引擎3-调试] 成功发现 {len(triggered_df)} 个共振信号点。开始准备数据库记录...")
        db_records = []
        for timestamp, row in triggered_df.iterrows():
            # ▼▼▼【代码修改】: 在此处合并所有剧本 ▼▼▼
            # 1. 获取分钟线自身的剧本
            resonance_playbook = resonance_params.get('signal_name', 'UNKNOWN_RESONANCE')
            
            # 2. 从row中提取当天生效的日线/周线剧本
            # row中包含了从日线df合并过来的 'playbook_*' 列
            daily_playbooks = [col.replace('playbook_', '') for col in row.index if col.startswith('playbook_') and row[col] is True]
            
            # 3. 合并所有剧本并去重
            combined_playbooks = list(set([resonance_playbook] + daily_playbooks))
            
            # 4. 创建基础记录，并用合并后的剧本覆盖默认值
            record = self._prepare_intraday_db_record(stock_code, timestamp, row, resonance_params)
            record['triggered_playbooks'] = combined_playbooks # 使用合并后的剧本列表
            # ▲▲▲【代码修改】: 结束 ▲▲▲

            # 计算融合分数
            daily_score = sanitize_for_json(row.get('daily_entry_score', 0.0))
            resonance_score = sanitize_for_json(resonance_params.get('score', 0.0))
            total_score = daily_score + resonance_score
            record['entry_score'] = total_score
            
            # print(f"      - [剧本与分数融合] 时间: {timestamp}, 日线分: {daily_score:.0f}, 分钟线基础分: {resonance_score:.0f}, 总分: {total_score:.0f}")
            # print(f"        - 合并后剧本: {combined_playbooks}")
            db_records.append(record)
        # print(f"    - [引擎3-调试] 数据库记录准备完成，共 {len(db_records)} 条。引擎正常结束。")
        return db_records

    def _run_intraday_take_profit_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        # ... 此方法保持不变 ...
        tp_params = self.tactical_config.get('strategy_params', {}).get('trend_follow', {}).get('intraday_take_profit_params', {})
        if not tp_params.get('enabled', False):
            return []
        tf = tp_params.get('timeframe')
        if not tf or tf not in all_dfs or all_dfs[tf].empty:
            # print(f"    - [止盈引擎] 结论: 止盈检查周期 '{tf}' 数据缺失或为空。引擎退出。")
            return []
        df = all_dfs[tf].copy()
        # print(f"    - [止盈引擎] 在 {tf} 分钟级别上检查 {len(tp_params.get('rules', []))} 条止盈规则...")
        all_exit_signals = pd.Series(False, index=df.index)
        exit_triggers = {}
        for rule in tp_params.get('rules', []):
            rule_type = rule.get('type')
            signal_code = rule.get('signal_code', 999)
            condition_signal = pd.Series(False, index=df.index)
            if rule_type == 'macd_dead_cross':
                p = rule['periods']
                macd_col = f'MACD_{p[0]}_{p[1]}_{p[2]}'
                signal_col = f'MACDs_{p[0]}_{p[1]}_{p[2]}'
                if macd_col in df.columns and signal_col in df.columns:
                    condition_signal = (df[macd_col] < df[signal_col]) & (df[macd_col].shift(1) >= df[signal_col].shift(1))
            elif rule_type == 'kdj_dead_cross':
                p = rule['periods']
                k_col, d_col = f'KDJk_{p[0]}_{p[1]}_{p[2]}', f'KDJd_{p[0]}_{p[1]}_{p[2]}'
                high_level = rule.get('high_level', 80)
                if k_col in df.columns and d_col in df.columns:
                    is_cross = (df[k_col] < df[d_col]) & (df[k_col].shift(1) >= df[d_col].shift(1))
                    is_in_zone = df[k_col].shift(1) > high_level
                    condition_signal = is_cross & is_in_zone
            elif rule_type == 'top_divergence':
                indicator = rule.get('indicator', 'rsi')
                if indicator == 'rsi':
                    p = rule['period'] # 注意这里是 period
                    rsi_col = f'RSI_{p}'
                    if 'close' in df.columns and rsi_col in df.columns:
                        peaks, _ = find_peaks(df['close'], distance=rule.get('lookback', 10))
                        if len(peaks) > 1:
                            divergence_points = pd.Series(False, index=df.index)
                            for i in range(1, len(peaks)):
                                prev_peak_idx, curr_peak_idx = peaks[i-1], peaks[i]
                                if (df['close'].iloc[curr_peak_idx] > df['close'].iloc[prev_peak_idx] and
                                    df[rsi_col].iloc[curr_peak_idx] < df[rsi_col].iloc[prev_peak_idx]):
                                    divergence_points.iloc[curr_peak_idx] = True
                            condition_signal = divergence_points
            if condition_signal.any():
                # print(f"      - [止盈引擎] 规则 '{rule_type}' (代码:{signal_code}) 触发 {condition_signal.sum()} 次。")
                all_exit_signals |= condition_signal
                exit_triggers[signal_code] = condition_signal
        triggered_df = df[all_exit_signals].copy()
        if triggered_df.empty:
            return []
        triggered_df['exit_signal_code'] = 0
        for code in sorted(exit_triggers.keys()):
            triggered_df.loc[exit_triggers[code], 'exit_signal_code'] = code
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
                "exit_signal_code": sanitize_for_json(row.get('exit_signal_code', 999)),
                "is_long_term_bullish": False,
                "is_mid_term_bullish": False,
                "is_pullback_setup": False,
                "pullback_target_price": None,
                "triggered_playbooks": [f"EXIT_CODE_{int(row.get('exit_signal_code', 999))}"],
                "context_snapshot": sanitize_for_json({'close': row.get('close')}),
            }
            db_records.append(record)
        return db_records
    
    def _check_single_condition(self, df: pd.DataFrame, cond: Dict, tf: str) -> pd.Series:
        # ... 此方法保持不变 ...
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
                hist_above_zero_strengthening = (df[hist_col] > 0) & (df[hist_col] > df[hist_col].shift(shift_periods)) & (df[hist_col].shift(shift_periods) < df[hist_col].shift(shift_periods * 2))
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
        # ... 此方法保持不变 ...
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
            "is_long_term_bullish": False,
            "is_mid_term_bullish": False,
            "is_pullback_setup": False,
            "pullback_target_price": None,
            "triggered_playbooks": [signal_name], # 注意：这个值将在调用此函数后被覆盖
            "context_snapshot": sanitize_for_json({'close': row.get('close')}),
        }
        return record
