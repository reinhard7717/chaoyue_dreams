# 文件: strategies/multi_timeframe_trend_strategy.py
# 版本: V6.0 - 日线/分钟线融合增强版

import asyncio
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import json
import logging
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from services.indicator_services import IndicatorService
from strategies.trend_following_strategy import TrendFollowStrategy
from strategies.weekly_trend_follow_strategy import WeeklyTrendFollowStrategy
from utils.config_loader import load_strategy_config
from utils.data_sanitizer import sanitize_for_json

logger = logging.getLogger(__name__)

class MultiTimeframeTrendStrategy:

    def __init__(self):
        """
        【V6.0 日线/分钟线融合增强版】
        - 核心逻辑升级: 分钟线信号的触发前提，从单一的“日线看涨”，升级为“日线看涨”或“当日为日线关键信号日”二者取其一。
        - 分数体系融合: 分钟线信号的最终得分，将是“日线策略得分”与“分钟线共振基础分”的总和，使信号质量评估更全面。
        - 数据流优化: 增加 self.daily_analysis_df 实例变量，用于在日线引擎和分钟线引擎之间传递完整的分析结果。
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

        final_indicators = self._merge_indicators(
            base_merged_fe_params.get('indicators', {}),
            resonance_indicators
        )
        base_merged_fe_params['indicators'] = final_indicators

        self.merged_config = deepcopy(self.tactical_config)
        self.merged_config['feature_engineering_params'] = base_merged_fe_params
        
        if 'strategy_playbooks' in self.strategic_config:
            self.merged_config['strategy_playbooks'] = deepcopy(self.strategic_config['strategy_playbooks'])
        
        self.indicator_service = IndicatorService()
        self.strategic_engine = WeeklyTrendFollowStrategy(config=self.strategic_config) 
        self.tactical_engine = TrendFollowStrategy(config=self.tactical_config)
        
        # ▼▼▼新增实例变量，用于在引擎间传递日线分析结果 ▼▼▼
        self.daily_analysis_df = None
        

        self.required_timeframes = self.indicator_service._discover_required_timeframes_from_config(self.merged_config)

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
        logger.info(f"--- 开始为【{stock_code}】执行三级引擎分析 (V6.0) ---")
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
        logger.info(f"--- 引擎2: 【战术引擎】运行完毕，生成 {len(tactical_records)} 条日线信号。 ---")
        logger.info(f"\n--- 引擎3: 开始运行【执行引擎】(分钟线)... ---")
        execution_records = self._run_intraday_resonance_engine(stock_code, all_dfs)
        logger.info(f"--- 引擎3: 【执行引擎】运行完毕，生成 {len(execution_records)} 条分钟线信号。 ---")
        all_records = tactical_records + execution_records
        logger.info(f"\n--- 【{stock_code}】所有引擎分析完成，共生成 {len(all_records)} 条信号记录。 ---")
        return all_records if all_records else None

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
        # ▼▼▼保存完整的日线分析结果以供分钟线引擎使用 ▼▼▼
        self.daily_analysis_df = final_df
        
        if final_df is None or final_df.empty: return []
        return self.tactical_engine.prepare_db_records(stock_code, final_df, atomic_signals, params=self.tactical_config, result_timeframe='D')

    def _run_intraday_resonance_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        print("\n--- [引擎3-调试 V6.0 日线/分钟线融合增强版] 进入 _run_intraday_resonance_engine ---")
        resonance_params = self.tactical_config.get('strategy_params', {}).get('trend_follow', {}).get('multi_level_resonance_params', {})
        if not resonance_params.get('enabled', False):
            print("    - [引擎3-调试] 结论: 分钟共振剧本在JSON中未启用 (enabled: false)。引擎退出。")
            return []
        
        # ▼▼▼检查日线分析结果是否存在 ▼▼▼
        if self.daily_analysis_df is None or self.daily_analysis_df.empty:
            print("    - [引擎3-调试] 结论: 缺少日线策略分析结果 (self.daily_analysis_df)，无法执行分钟线引擎。引擎退出。")
            return []
        

        levels = resonance_params.get('levels', [])
        if not levels:
            print("    - [引擎3-调试] 结论: JSON中未定义任何共振层级 (levels为空)。引擎退出。")
            return []
        trigger_tf = levels[-1]['tf']
        if trigger_tf not in all_dfs or all_dfs[trigger_tf].empty:
            print(f"    - [引擎3-调试] 结论: 缺少或为空的触发周期 '{trigger_tf}' 数据。引擎退出。")
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
                print(f"    - [引擎3-调试] 结论: 共振检查缺少关键的上层周期 '{level_tf}' 数据。引擎退出。")
                return []
        
        # ▼▼▼融合日线趋势和得分，创建新的、更全面的过滤器 ▼▼▼
        print("    - [引擎3-调试] 新增步骤: 融合日线趋势与得分，创建综合过滤器...")
        # 从日线分析结果中提取关键列
        daily_score_threshold = self.tactical_config.get('entry_scoring_params', {}).get('score_threshold', 100)
        daily_context_df = self.daily_analysis_df[['context_mid_term_bullish', 'entry_score']].copy()
        
        # 条件A: 日线趋势已看涨
        is_bullish_trend = daily_context_df['context_mid_term_bullish']
        # 条件B: 当日是日线级别的关键信号日（得分足够高）
        is_reversal_day = daily_context_df['entry_score'] >= daily_score_threshold
        
        # 最终前提条件：满足A或B即可
        daily_context_df['is_daily_trend_ok'] = is_bullish_trend | is_reversal_day
        
        # 为了分数融合，重命名entry_score以避免冲突
        daily_context_df.rename(columns={'entry_score': 'daily_entry_score'}, inplace=True)
        
        print(f"      - [引擎3-调试] 日线看涨天数: {is_bullish_trend.sum()}, 日线信号日天数: {is_reversal_day.sum()}, 总合格天数: {daily_context_df['is_daily_trend_ok'].sum()}")

        # 将日线综合状态合并到分钟线对齐后的DataFrame中
        df_aligned = pd.merge_asof(
            left=df_aligned,
            right=daily_context_df[['is_daily_trend_ok', 'daily_entry_score']],
            left_index=True,
            right_index=True,
            direction='backward'
        )
        df_aligned['is_daily_trend_ok'].fillna(False, inplace=True)
        df_aligned['daily_entry_score'].fillna(0, inplace=True)
        
        print("    - [引擎3-调试] 数据对齐完成。开始逐级检查共振条件...")
        final_signal = pd.Series(True, index=df_aligned.index)

        # 应用新的日线综合过滤器
        final_signal &= df_aligned['is_daily_trend_ok']
        print(f"    - [引擎3-调试] 应用日线综合过滤后，剩余候选信号点: {final_signal.sum()}")
        if final_signal.sum() == 0:
            print("    - [引擎3-调试] 结论: 所有时间点均不满足日线前提，无需进一步检查分钟线条件。引擎正常结束。")
            return []
        

        for i, level in enumerate(levels):
            level_tf, level_name = level['tf'], level.get('level_name', f'Level_{i}')
            level_logic, level_conditions = level.get('logic', 'AND').upper(), level.get('conditions', [])
            print(f"      - [引擎3-调试] 正在检查第 {i+1} 层: '{level_name}' (周期: {level_tf}, 逻辑: {level_logic})")
            level_signal = pd.Series(True if level_logic == 'AND' else False, index=df_aligned.index)
            for cond in level_conditions:
                cond_signal = self._check_single_condition(df_aligned, cond, level_tf)
                if level_logic == 'AND': level_signal &= cond_signal
                else: level_signal |= cond_signal
            final_signal &= level_signal
        
        print(f"    - [引擎3-调试] 所有层级检查完毕。最终共振信号触发总次数: {final_signal.sum()}")
        triggered_df = df_aligned[final_signal]
        if triggered_df.empty:
            print("    - [引擎3-调试] 结论: 没有发现任何满足所有条件的共振信号点。引擎正常结束。")
            return []
        
        print(f"    - [引擎3-调试] 成功发现 {len(triggered_df)} 个共振信号点。开始准备数据库记录...")
        db_records = []
        for timestamp, row in triggered_df.iterrows():
            record = self._prepare_intraday_db_record(stock_code, timestamp, row, resonance_params)
            
            # ▼▼▼重新计算并覆盖得分为“日线得分 + 分钟线基础分” ▼▼▼
            daily_score = sanitize_for_json(row.get('daily_entry_score', 0.0))
            resonance_score = sanitize_for_json(resonance_params.get('score', 0.0))
            total_score = daily_score + resonance_score
            record['entry_score'] = total_score
            print(f"      - [分数融合] 时间: {timestamp}, 日线分: {daily_score:.0f}, 分钟线基础分: {resonance_score:.0f}, 总分: {total_score:.0f}")
            
            
            db_records.append(record)
        print(f"    - [引擎3-调试] 数据库记录准备完成，共 {len(db_records)} 条。引擎正常结束。")
        return db_records

    def _check_single_condition(self, df: pd.DataFrame, cond: Dict, tf: str) -> pd.Series:
        """
        【V5.9 信号逻辑增强版】
        - 核心升级: 彻底重构了分钟线信号的判断逻辑，使其能够捕捉更多类型的反转。
        - 1. [增强] `rsi_reversal`: 不再局限于从超卖区上穿，增加了对“RSI在任意位置拐头向上”的判断，使其对趋势中的回调反转更敏感。
        - 2. [新增] `kdj_j_reversal`: 引入了更灵敏的J线拐头信号，作为股价启动的早期预警。
        - 3. [保留] `kdj_cross`: 保留了原有的低位金叉逻辑，作为备用。
        - 4. [新增] `macd_above_zero`: 增加了对MACD柱状图（hist）在0轴上方的二次走强的判断，用于捕捉趋势加速。
        """
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
                # print(f"          - [诊断-失败] 条件 '{cond_type}' 失败: 缺少列 {missing_cols}")
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
                # 原始条件：MACD线在0轴上方，且仍在上升
                is_above_zero_and_rising = (df[macd_line_col] > 0) & (df[macd_line_col] > df[macd_line_col].shift(shift_periods))
                # 新增条件：MACD柱状图在0轴上方，且出现收缩后的再次放大（二次走强）
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
        
        elif cond_type == 'kdj_cross': # 保留原有逻辑，用于稳健型低位金叉
            p = cond['periods']
            k_col, d_col = f'KDJk_{p[0]}_{p[1]}_{p[2]}{suffix}', f'KDJd_{p[0]}_{p[1]}_{p[2]}{suffix}'
            oversold_level = cond.get('low_level', 50)
            if check_cols(k_col, d_col):
                is_cross = (df[k_col] > df[d_col]) & (df[k_col].shift(shift_periods) <= df[d_col].shift(shift_periods))
                is_in_zone = df[d_col] < oversold_level
                return is_cross & is_in_zone
        
        # 【新增信号类型】
        elif cond_type == 'kdj_j_reversal':
            p = cond['periods']
            j_col = f'KDJj_{p[0]}_{p[1]}_{p[2]}{suffix}'
            low_level = cond.get('low_level', 30) # J线拐头的阈值可以设得更低
            if check_cols(j_col):
                # J线从低位拐头向上
                is_turning_up = (df[j_col] > df[j_col].shift(shift_periods))
                was_in_low_zone = (df[j_col].shift(shift_periods) < low_level)
                return is_turning_up & was_in_low_zone

        # 【增强信号类型】
        elif cond_type == 'rsi_reversal':
            p = cond['period']
            rsi_col = f'RSI_{p}{suffix}'
            oversold_level = cond.get('oversold_level', 35)
            if check_cols(rsi_col):
                # 原始逻辑：从超卖区上穿
                classic_reversal = (df[rsi_col] > oversold_level) & (df[rsi_col].shift(shift_periods) <= oversold_level)
                # 新增逻辑：RSI在任意位置，结束回调并拐头向上
                # 定义“回调结束”为：前一刻RSI在下跌，此刻RSI在上涨
                is_turning_up_after_dip = (df[rsi_col] > df[rsi_col].shift(shift_periods)) & \
                                          (df[rsi_col].shift(shift_periods) < df[rsi_col].shift(shift_periods * 2))
                return classic_reversal | is_turning_up_after_dip
        
        return pd.Series(False, index=df.index)

    def _prepare_intraday_db_record(self, stock_code: str, timestamp: pd.Timestamp, row: pd.Series, params: dict) -> Dict[str, Any]:
        signal_name = params.get('signal_name', 'UNKNOWN_RESONANCE')
        trigger_tf = params['levels'][-1]['tf']
        native_utc_datetime: datetime = timestamp.to_pydatetime()
        
        # print(f"    - [数据准备] 准备数据库记录，已转换为原生datetime对象: {native_utc_datetime} (类型: {type(native_utc_datetime)})")

        record = {
            "stock_code": stock_code,
            "trade_time": native_utc_datetime, # 修改: 使用转换后的原生datetime对象
            "timeframe": trigger_tf,
            "strategy_name": signal_name,
            "close_price": sanitize_for_json(row.get('close')),
            "entry_score": sanitize_for_json(params.get('score', 0.0)), # 注意：这个分数将在外部被覆盖
            "entry_signal": True,
            "exit_signal_code": 0,
            "is_long_term_bullish": False,
            "is_mid_term_bullish": False,
            "is_pullback_setup": False,
            "pullback_target_price": None,
            "triggered_playbooks": [signal_name],
            "context_snapshot": sanitize_for_json({'close': row.get('close')}),
        }
        return record
