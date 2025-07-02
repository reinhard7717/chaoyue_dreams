# 文件: strategies/multi_timeframe_trend_strategy.py
# 版本: V5.8 - 强制重命名终极修复版

import asyncio
from collections import defaultdict
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
        【V5.8 强制重命名终极修复版】
        - 核心BUG修复: 彻底解决了因错误理解 merge_asof 的 suffixes 参数，导致高频周期指标列名未被正确添加后缀的根本问题。
        - 解决方案: 在 _run_intraday_resonance_engine 方法中，不再依赖 suffixes 参数。
                   改为在合并前，手动、强制地为右侧DataFrame的所有列添加后缀（例如 'close' -> 'close_60'）。
                   这确保了所有指标列在合并后的DataFrame中都拥有唯一且可预测的名称，使得后续的条件检查和“智能移位”逻辑能够命中正确的数据。
                   这是本次调试的最终修复。
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
        logger.info(f"--- 开始为【{stock_code}】执行三级引擎分析 (V5.8) ---")
        logger.info(f"--- 准备阶段: 调用 IndicatorService 统一准备所有数据... ---")
        all_dfs = await self.indicator_service._prepare_base_data_and_indicators(stock_code, self.merged_config, trade_time)
        if not all_dfs or 'D' not in all_dfs or 'W' not in all_dfs:
            logger.warning(f"[{stock_code}] 核心数据(周线或日线)准备失败，分析终止。")
            return None
        logger.info("--- [数据标准化] 开始统一所有DataFrame的索引为UTC时区... ---")
        for key, df in all_dfs.items():
            if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex): continue
            if df.index.tz is None: 
                # 核心修复：先将无时区的中国时间本地化为'Asia/Shanghai'，然后再转换为标准的UTC时间
                # 旧的错误代码: all_dfs[key].index = df.index.tz_localize('UTC')
                all_dfs[key].index = df.index.tz_localize('Asia/Shanghai').tz_convert('UTC')
            elif str(df.index.tz) != 'UTC': all_dfs[key].index = df.index.tz_convert('UTC')
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
        if final_df is None or final_df.empty: return []
        return self.tactical_engine.prepare_db_records(stock_code, final_df, atomic_signals, params=self.tactical_config, result_timeframe='D')

    def _run_intraday_resonance_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        print("\n--- [引擎3-调试 V5.8] 进入 _run_intraday_resonance_engine ---")
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
        trigger_tf = levels[-1]['tf']
        if trigger_tf not in all_dfs or all_dfs[trigger_tf].empty:
            print(f"    - [引擎3-调试] 结论: 缺少或为空的触发周期 '{trigger_tf}' 数据。引擎退出。")
            return []
        print(f"    - [引擎3-调试] 检查通过: 触发周期 '{trigger_tf}' 数据存在，行数: {len(all_dfs[trigger_tf])}。")
        print("    - [引擎3-调试] 开始使用【强制重命名】逻辑对齐所有周期数据...")
        df_aligned = all_dfs[trigger_tf].copy()
        for level in levels[:-1]:
            level_tf = level['tf']
            if level_tf in all_dfs and not all_dfs[level_tf].empty:
                print(f"      - [引擎3-调试] 正在合并周期 '{level_tf}' (行数: {len(all_dfs[level_tf])}) 的数据...")
                
                # 【核心修复】: 强制为右侧DataFrame的所有列添加后缀
                df_right = all_dfs[level_tf].copy()
                rename_map = {col: f"{col}_{level_tf}" for col in df_right.columns}
                df_right.rename(columns=rename_map, inplace=True)
                
                # 现在进行合并，不再需要suffixes参数，因为列名已保证唯一
                df_aligned = pd.merge_asof(
                    left=df_aligned, right=df_right, left_index=True, right_index=True, direction='backward'
                )
                print(f"      - [引擎3-调试] 合并后，对齐后的DataFrame行数: {len(df_aligned)}")
            else:
                print(f"    - [引擎3-调试] 结论: 共振检查缺少关键的上层周期 '{level_tf}' 数据。引擎退出。")
                return []
        print("    - [引擎3-调试] 数据对齐完成。开始逐级检查共振条件...")
        final_signal = pd.Series(True, index=df_aligned.index)
        for i, level in enumerate(levels):
            level_tf, level_name = level['tf'], level.get('level_name', f'Level_{i}')
            level_logic, level_conditions = level.get('logic', 'AND').upper(), level.get('conditions', [])
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
        record = {
            "stock_code": stock_code, "trade_time": sanitize_for_json(timestamp), "timeframe": trigger_tf,
            "strategy_name": signal_name, "close_price": sanitize_for_json(row.get('close')),
            "entry_score": sanitize_for_json(params.get('score', 0.0)), "entry_signal": True, "exit_signal_code": 0,
            "is_long_term_bullish": False, "is_mid_term_bullish": False, "is_pullback_setup": False,
            "pullback_target_price": None, "triggered_playbooks": [signal_name],
            "context_snapshot": sanitize_for_json({'close': row.get('close')}),
        }
        return record
