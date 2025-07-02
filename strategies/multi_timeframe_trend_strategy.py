# 文件: strategies/multi_timeframe_trend_strategy.py
# 版本: V5.4 - 共振引擎逻辑修复版

import asyncio
from copy import deepcopy
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
        【V5.4 共振引擎逻辑修复版】
        - 核心升级: 重写了 _check_single_condition 方法，修复了多个共振条件中过于严苛或错误的逻辑，
                   并增加了详细的诊断日志，从根本上解决共振信号为0的问题。
        """
        tactical_config_path = 'config/trend_follow_strategy.json'
        strategic_config_path = 'config/weekly_trend_follow_strategy.json'
        
        self.tactical_config = load_strategy_config(tactical_config_path)
        self.strategic_config = load_strategy_config(strategic_config_path)

        merged_fe_params = self._merge_feature_engineering_configs(
            self.tactical_config.get('feature_engineering_params', {}),
            self.strategic_config.get('feature_engineering_params', {})
        )

        self.merged_config = deepcopy(self.tactical_config)
        self.merged_config['feature_engineering_params'] = merged_fe_params
        
        if 'strategy_playbooks' in self.strategic_config:
            self.merged_config['strategy_playbooks'] = deepcopy(self.strategic_config['strategy_playbooks'])

        self.indicator_service = IndicatorService()
        self.strategic_engine = WeeklyTrendFollowStrategy(config=self.strategic_config) 
        self.tactical_engine = TrendFollowStrategy(config=self.tactical_config)

        self.required_timeframes = self.indicator_service._discover_required_timeframes_from_config(self.merged_config)

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

    def _merge_indicators(self, tactical_indicators, strategic_indicators):
        merged = {}
        all_keys = set(tactical_indicators.keys()) | set(strategic_indicators.keys())

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
            tactical_cfg = tactical_indicators.get(key, {})
            strategic_cfg = strategic_indicators.get(key, {})
            is_enabled = tactical_cfg.get('enabled', False) or strategic_cfg.get('enabled', False)
            if not is_enabled: continue
            tactical_sub_configs = standardize_to_configs(tactical_cfg)
            strategic_sub_configs = standardize_to_configs(strategic_cfg)
            final_configs = tactical_sub_configs + strategic_sub_configs
            if not final_configs:
                if key in tactical_cfg or key in strategic_cfg:
                     merged[key] = deepcopy(tactical_cfg)
                     merged[key].update(deepcopy(strategic_cfg))
                continue
            merged[key] = {
                'enabled': True,
                '说明': tactical_cfg.get('说明', '') or strategic_cfg.get('说明', ''),
                'configs': final_configs
            }
            if not final_configs and 'enabled' in (tactical_cfg or strategic_cfg):
                 merged[key] = {'enabled': is_enabled, '说明': merged[key]['说明']}
        return merged

    async def run_for_stock(self, stock_code: str, trade_time: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        logger.info(f"--- 开始为【{stock_code}】执行三级引擎分析 (V5.4) ---")
        logger.info(f"--- 准备阶段: 调用 IndicatorService 统一准备所有数据... ---")
        all_dfs = await self.indicator_service._prepare_base_data_and_indicators(
            stock_code, self.merged_config, trade_time
        )
        if not all_dfs or 'D' not in all_dfs or 'W' not in all_dfs:
            logger.warning(f"[{stock_code}] 核心数据(周线或日线)准备失败，分析终止。")
            return None
        
        logger.info("--- [数据标准化] 开始统一所有DataFrame的索引为UTC时区... ---")
        for key, df in all_dfs.items():
            if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex): continue
            if df.index.tz is None:
                all_dfs[key].index = df.index.tz_localize('UTC')
            elif str(df.index.tz) != 'UTC':
                all_dfs[key].index = df.index.tz_convert('UTC')
        logger.info("--- [数据标准化] 所有索引已统一为UTC时区。 ---")

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
        df_merged = pd.merge_asof(
            left=df_daily_copy.sort_index(),
            right=strategic_signals_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )
        for col in strategic_signals_df.columns:
            if col not in df_merged.columns: continue
            if col.startswith(('playbook_', 'signal_', 'state_', 'event_', 'filter_')):
                if col == 'signal_breakout_trigger_W':
                    new_col_name = 'BASE_SIGNAL_BREAKOUT_TRIGGER'
                    df_merged.rename(columns={col: new_col_name}, inplace=True)
                    df_merged[new_col_name] = df_merged[new_col_name].fillna(False).astype(bool)
                else:
                    df_merged[col] = df_merged[col].fillna(False).astype(bool)
            elif col.startswith(('washout_score_', 'rejection_signal_')):
                df_merged[col] = df_merged[col].fillna(0).astype(int)
        return df_merged
    
    def _run_tactical_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        final_df, atomic_signals = self.tactical_engine.apply_strategy(all_dfs, self.tactical_config)
        if final_df is None or final_df.empty: return []
        db_records = self.tactical_engine.prepare_db_records(
            stock_code, final_df, atomic_signals, params=self.tactical_config, result_timeframe='D'
        )
        return db_records

    def _run_intraday_resonance_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        print("\n--- [引擎3-调试] 进入 _run_intraday_resonance_engine ---")
        resonance_params = self.tactical_config.get('strategy_params', {}).get('trend_follow', {}).get('multi_level_resonance_params', {})
        if not resonance_params.get('enabled', False):
            print("    - [引擎3-调试] 结论: 分钟共振剧本在JSON中未启用 (enabled: false)。引擎退出。")
            return []
        print("    - [引擎3-调试] 检查通过: 剧本已启用。")
        levels = resonance_params.get('levels', [])
        if not levels:
            print("    - [引擎3-调试] 结论: JSON中未定义任何共振层级 (levels为空)。引擎退出。")
            return []
        print(f"    - [引擎3-调试] 检查通过: 共找到 {len(levels)} 个共振层级。")
        trigger_level = levels[-1]
        trigger_tf = trigger_level['tf']
        if trigger_tf not in all_dfs or all_dfs[trigger_tf].empty:
            print(f"    - [引擎3-调试] 结论: 缺少或为空的触发周期 '{trigger_tf}' 数据。引擎退出。")
            return []
        print(f"    - [引擎3-调试] 检查通过: 触发周期 '{trigger_tf}' 数据存在，行数: {len(all_dfs[trigger_tf])}。")
        print("    - [引擎3-调试] 开始将所有上层周期数据对齐到触发周期...")
        df_aligned = all_dfs[trigger_tf].copy()
        for level in levels[:-1]:
            level_tf = level['tf']
            if level_tf in all_dfs and not all_dfs[level_tf].empty:
                print(f"      - [引擎3-调试] 正在合并周期 '{level_tf}' (行数: {len(all_dfs[level_tf])}) 的数据...")
                df_aligned = pd.merge_asof(
                    left=df_aligned, right=all_dfs[level_tf], left_index=True, right_index=True,
                    direction='backward', suffixes=('', f'_{level_tf}')
                )
                print(f"      - [引擎3-调试] 合并后，对齐后的DataFrame行数: {len(df_aligned)}")
            else:
                print(f"    - [引擎3-调试] 结论: 共振检查缺少关键的上层周期 '{level_tf}' 数据。引擎退出。")
                return []
        print("    - [引擎3-调试] 数据对齐完成。开始逐级检查共振条件...")
        final_signal = pd.Series(True, index=df_aligned.index)
        for i, level in enumerate(levels):
            level_tf = level['tf']
            level_name = level.get('level_name', f'Level_{i}')
            level_logic = level.get('logic', 'AND').upper()
            level_conditions = level.get('conditions', [])
            print(f"      - [引擎3-调试] 正在检查第 {i+1} 层: '{level_name}' (周期: {level_tf}, 逻辑: {level_logic})")
            level_signal = pd.Series(True if level_logic == 'AND' else False, index=df_aligned.index)
            for cond in level_conditions:
                cond_signal = self._check_single_condition(df_aligned, cond, level_tf)
                print(f"        - [引擎3-调试] 条件 '{cond['type']}' 在周期 '{level_tf}' 上触发了 {cond_signal.sum()} 次。")
                if level_logic == 'AND': level_signal &= cond_signal
                else: level_signal |= cond_signal
            print(f"      - [引擎3-调试] 第 {i+1} 层总计触发 {level_signal.sum()} 次。")
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
            db_records.append(record)
        print(f"    - [引擎3-调试] 数据库记录准备完成，共 {len(db_records)} 条。引擎正常结束。")
        return db_records

    def _check_single_condition(self, df: pd.DataFrame, cond: Dict, tf: str) -> pd.Series:
        """
        【V2.0 逻辑修复 & 诊断增强版】
        辅助函数：检查单个共振条件并返回布尔序列。
        - 修复了多个条件的逻辑错误。
        - 增加了详细的列名检查日志。
        """
        cond_type = cond['type']
        # 确定列名后缀，只有非触发周期的列才有后缀
        trigger_tf = self.tactical_config.get('strategy_params', {}).get('trend_follow', {}).get('multi_level_resonance_params', {}).get('levels', [{}])[-1].get('tf')
        suffix = f'_{tf}' if tf != trigger_tf else ''
        
        # 内部辅助函数，用于检查列是否存在并打印日志
        def check_cols(*cols):
            for col in cols:
                if col not in df.columns:
                    print(f"          - [诊断-失败] 条件 '{cond_type}' 失败: 缺少列 '{col}'")
                    return False
            # print(f"          - [诊断-成功] 条件 '{cond_type}' 需要的列 {list(cols)} 全部存在。")
            return True

        # --- 开始检查各种条件类型 ---
        if cond_type == 'ema_above':
            period = cond['period']
            ema_col, close_col = f'EMA_{period}{suffix}', f'close{suffix}'
            if check_cols(ema_col, close_col):
                return df[close_col] > df[ema_col]

        elif cond_type == 'macd_above_zero':
            p = cond['periods']
            # 【逻辑修复】: 检查MACD快线(MACD)是否>0，而不是柱状线(MACDh)
            macd_line_col = f'MACD_{p[0]}_{p[1]}_{p[2]}{suffix}'
            if check_cols(macd_line_col):
                return df[macd_line_col] > 0

        elif cond_type == 'macd_cross':
            p = cond['periods']
            hist_col = f'MACDh_{p[0]}_{p[1]}_{p[2]}{suffix}'
            if check_cols(hist_col):
                return (df[hist_col] > 0) & (df[hist_col].shift(1) <= 0)

        elif cond_type == 'macd_hist_turning_up':
            p = cond['periods']
            hist_col = f'MACDh_{p[0]}_{p[1]}_{p[2]}{suffix}'
            if check_cols(hist_col):
                # 【逻辑修复】: 只检查柱状线是否增长，无论其在0轴上方或下方
                return df[hist_col] > df[hist_col].shift(1)

        elif cond_type == 'dmi_cross':
            p = cond['period']
            pdi_col, mdi_col = f'DMP_{p}{suffix}', f'DMN_{p}{suffix}'
            if check_cols(pdi_col, mdi_col):
                return (df[pdi_col] > df[mdi_col]) & (df[pdi_col].shift(1) <= df[mdi_col].shift(1))

        elif cond_type == 'kdj_cross':
            p = cond['periods']
            k_col, d_col = f'KDJk_{p[0]}_{p[1]}_{p[2]}{suffix}', f'KDJd_{p[0]}_{p[1]}_{p[2]}{suffix}'
            oversold_level = cond.get('low_level', 20) # 使用更通用的名称和默认值
            if check_cols(k_col, d_col):
                is_cross = (df[k_col] > df[d_col]) & (df[k_col].shift(1) <= df[d_col].shift(1))
                # 【逻辑修复】: 检查D线是否在超卖区，这比检查K线更可靠
                is_oversold = df[d_col] < oversold_level
                return is_cross & is_oversold

        elif cond_type == 'rsi_reversal':
            p = cond['period']
            rsi_col = f'RSI_{p}{suffix}'
            oversold_level = cond.get('oversold_level', 30)
            if check_cols(rsi_col):
                return (df[rsi_col] > oversold_level) & (df[rsi_col].shift(1) <= oversold_level)
        
        # 如果条件类型未知或缺少列，返回全False序列
        return pd.Series(False, index=df.index)

    def _prepare_intraday_db_record(self, stock_code: str, timestamp: pd.Timestamp, row: pd.Series, params: dict) -> Dict[str, Any]:
        signal_name = params.get('signal_name', 'UNKNOWN_RESONANCE')
        trigger_tf = params['levels'][-1]['tf']
        record = {
            "stock_code": stock_code,
            "trade_time": sanitize_for_json(timestamp),
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
            "triggered_playbooks": [signal_name],
            "context_snapshot": sanitize_for_json({'close': row.get('close')}),
        }
        return record
