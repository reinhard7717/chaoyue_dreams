# 文件: strategies/multi_timeframe_trend_strategy.py
# 版本: V4.0 - 双引擎并行版 (日线决策 + 分钟共振)
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
from utils.data_sanitizer import sanitize_for_json # 导入数据清洗工具

logger = logging.getLogger(__name__)

class MultiTimeframeTrendStrategy:

    def __init__(self):
        """
        【V5.3 健壮合并版】
        - 核心升级: 重写了 _merge_indicators 函数，采用更健壮、一致且无损的合并逻辑。
        - 解决方案: 新逻辑确保无论是哪个配置文件中的指标，无论是何种配置格式（简单或configs列表），
                   都会被统一、正确地标准化并合并到最终的 merged_config 中，从根源上解决了因配置合并不当导致的“缺少列”问题。
        """
        # 加载各自的配置文件
        tactical_config_path = 'config/trend_follow_strategy.json'
        strategic_config_path = 'config/weekly_trend_follow_strategy.json'
        
        self.tactical_config = load_strategy_config(tactical_config_path)
        self.strategic_config = load_strategy_config(strategic_config_path)

        # ▼▼▼ 使用全新的、健壮的合并逻辑 ▼▼▼
        
        # 步骤1: 智能合并数据工程层的配置
        merged_fe_params = self._merge_feature_engineering_configs(
            self.tactical_config.get('feature_engineering_params', {}),
            self.strategic_config.get('feature_engineering_params', {})
        )

        # 步骤2: 构建最终的合并后配置
        self.merged_config = deepcopy(self.tactical_config)
        self.merged_config['feature_engineering_params'] = merged_fe_params
        
        if 'strategy_playbooks' in self.strategic_config:
            self.merged_config['strategy_playbooks'] = deepcopy(self.strategic_config['strategy_playbooks'])

        # 初始化服务和引擎
        self.indicator_service = IndicatorService()
        self.strategic_engine = WeeklyTrendFollowStrategy(config=self.strategic_config) 
        self.tactical_engine = TrendFollowStrategy(config=self.tactical_config)

        # 周期发现逻辑基于【健壮合并后】的完整配置来运行
        self.required_timeframes = self.indicator_service._discover_required_timeframes_from_config(self.merged_config)
        
        # print(f"--- [总指挥 MultiTimeframeTrendStrategy (V5.3)] 初始化完成 ---")
        # print(f"    - [总指挥] 已通过【健壮合并】识别出所有必需周期: {sorted(list(self.required_timeframes))}")
        # 打印合并后的部分关键指标配置以供调试
        # print("    - [调试] 合并后 TRIX 配置:", json.dumps(self.merged_config['feature_engineering_params']['indicators'].get('trix', {}), indent=2, ensure_ascii=False))
        # print("    - [调试] 合并后 Coppock 配置:", json.dumps(self.merged_config['feature_engineering_params']['indicators'].get('coppock', {}), indent=2, ensure_ascii=False))
        # print("    - [调试] 合并后 EMA 配置:", json.dumps(self.merged_config['feature_engineering_params']['indicators'].get('ema', {}), indent=2, ensure_ascii=False))

    def _merge_feature_engineering_configs(self, tactical_fe, strategic_fe):
        """
        专门用于合并 feature_engineering_params 的辅助函数。(V5.3 无变化)
        """
        merged = deepcopy(tactical_fe)
        merged['base_needed_bars'] = max(
            tactical_fe.get('base_needed_bars', 0),
            strategic_fe.get('base_needed_bars', 0)
        )
        
        # ▼▼▼ 调用全新重写的 _merge_indicators 函数 ▼▼▼
        merged['indicators'] = self._merge_indicators(
            tactical_fe.get('indicators', {}),
            strategic_fe.get('indicators', {})
        )
        
        
        return merged

    def _merge_indicators(self, tactical_indicators, strategic_indicators):
        """
        【V5.3 健壮合并版】智能、一致且无损地合并两个指标配置字典。
        """
        merged = {}
        # 获取两个配置中所有指标键的并集，确保一个不漏
        all_keys = set(tactical_indicators.keys()) | set(strategic_indicators.keys())

        def standardize_to_configs(cfg):
            """辅助函数：将任何格式的指标配置标准化为 configs 列表格式。"""
            if not cfg or not cfg.get('enabled', False):
                return []
            
            # 如果已经是 configs 格式，直接返回
            if 'configs' in cfg:
                return deepcopy(cfg['configs'])
            
            # 如果是简单格式，转换为 configs 列表
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

            # 只要任意一个配置启用，最终就启用
            is_enabled = tactical_cfg.get('enabled', False) or strategic_cfg.get('enabled', False)
            if not is_enabled:
                continue

            # 标准化两个配置，得到它们的 configs 列表
            tactical_sub_configs = standardize_to_configs(tactical_cfg)
            strategic_sub_configs = standardize_to_configs(strategic_cfg)
            
            # 拼接两个列表，形成完整的需求清单
            final_configs = tactical_sub_configs + strategic_sub_configs
            
            if not final_configs:
                # 如果合并后没有有效的配置项（例如，都未启用或格式错误），则跳过
                # 但保留其他顶层键，如 cyq_perf 的 enabled
                if key in tactical_cfg or key in strategic_cfg:
                     merged[key] = deepcopy(tactical_cfg)
                     merged[key].update(deepcopy(strategic_cfg))
                continue

            # 构建最终的、干净的合并后配置
            merged[key] = {
                'enabled': True,
                # 保留任一配置中的说明
                '说明': tactical_cfg.get('说明', '') or strategic_cfg.get('说明', ''),
                'configs': final_configs
            }
            # 特殊处理那些没有 configs 列表的配置，比如 cyq_perf
            if not final_configs and 'enabled' in (tactical_cfg or strategic_cfg):
                 merged[key] = {'enabled': is_enabled, '说明': merged[key]['说明']}


        return merged

    async def run_for_stock(self, stock_code: str, trade_time: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        【V5.0 三级引擎版】为单个股票执行战略、战术、执行三级分析。
        """
        logger.info(f"--- 开始为【{stock_code}】执行三级引擎分析 (V5.0) ---")

        # --- 准备阶段: 统一获取所有周期数据 ---
        logger.info(f"--- 准备阶段: 调用 IndicatorService 统一准备所有数据... ---")
        all_dfs = await self.indicator_service._prepare_base_data_and_indicators(
            stock_code, self.merged_config, trade_time
        )

        if not all_dfs or 'D' not in all_dfs or 'W' not in all_dfs:
            logger.warning(f"[{stock_code}] 核心数据(周线或日线)准备失败，分析终止。")
            return None
        
        # ▼▼▼ 全局时区标准化步骤，这是本次修复的核心 ▼▼▼
        logger.info("--- [数据标准化] 开始统一所有DataFrame的索引为UTC时区... ---")
        for key, df in all_dfs.items():
            if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
                continue
            
            # 检查并统一时区
            if df.index.tz is None:
                # 如果索引是“天真”的 (naive)，则本地化为UTC
                all_dfs[key].index = df.index.tz_localize('UTC')
                # print(f"    - [标准化] 周期 '{key}' 的索引已从 naive 本地化为 UTC。")
            elif str(df.index.tz) != 'UTC':
                # 如果索引有时区但不是UTC，则转换为UTC
                all_dfs[key].index = df.index.tz_convert('UTC')
                # print(f"    - [标准化] 周期 '{key}' 的索引已从 {df.index.tz} 转换为 UTC。")
        logger.info("--- [数据标准化] 所有索引已统一为UTC时区。 ---")

        if 'D' not in all_dfs or 'W' not in all_dfs:
            logger.warning(f"[{stock_code}] 核心数据(周线或日线)准备失败，分析终止。")
            return None
        
        # --- 引擎 1: 运行【战略引擎】(周线) ---
        logger.info(f"\n--- 引擎1: 开始运行【战略引擎】(周线)... ---")
        strategic_signals_df = self._run_strategic_engine(all_dfs['W'])
        logger.info(f"--- 引擎1: 【战略引擎】运行完毕。---")

        # --- 数据流转: 将战略信号整合到日线数据中 ---
        logger.info(f"\n--- 数据流转: 整合战略信号到日线数据... ---")
        all_dfs['D'] = self._merge_strategic_signals_to_daily(all_dfs['D'], strategic_signals_df)
        
        # --- 引擎 2: 运行【战术引擎】(日线) ---
        logger.info(f"\n--- 引擎2: 开始运行【战术引擎】(日线)... ---")
        tactical_records = self._run_tactical_engine(stock_code, all_dfs)
        logger.info(f"--- 引擎2: 【战术引擎】运行完毕，生成 {len(tactical_records)} 条日线信号。 ---")

        # --- 引擎 3: 运行【执行引擎】(分钟线) ---
        logger.info(f"\n--- 引擎3: 开始运行【执行引擎】(分钟线)... ---")
        execution_records = self._run_intraday_resonance_engine(stock_code, all_dfs)
        logger.info(f"--- 引擎3: 【执行引擎】运行完毕，生成 {len(execution_records)} 条分钟线信号。 ---")

        # --- 汇总阶段: 合并所有结果并返回 ---
        all_records = tactical_records + execution_records
        logger.info(f"\n--- 【{stock_code}】所有引擎分析完成，共生成 {len(all_records)} 条信号记录。 ---")
        return all_records if all_records else None

    def _run_strategic_engine(self, df_weekly: pd.DataFrame) -> pd.DataFrame:
        """【引擎1】执行周线级别的战略分析。"""
        if df_weekly is None or df_weekly.empty:
            logger.warning("周线数据为空，战略引擎跳过。")
            return pd.DataFrame()
        
        # 调用周线策略实例的 apply_strategy 方法
        return self.strategic_engine.apply_strategy(df_weekly)
    
    def _merge_strategic_signals_to_daily(self, df_daily: pd.DataFrame, strategic_signals_df: pd.DataFrame) -> pd.DataFrame:
        """将周线战略信号合并到日线DataFrame中。"""
        if strategic_signals_df is None or strategic_signals_df.empty:
            return df_daily

        df_daily_copy = df_daily.copy()

        # 使用 merge_asof 将周线信号向下广播到日线
        df_merged = pd.merge_asof(
            left=df_daily_copy.sort_index(),
            right=strategic_signals_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )
        
        # print("    - [数据流转] 开始对所有合并的周线信号列进行全面NaN清洗...")
        for col in strategic_signals_df.columns:
            if col not in df_merged.columns:
                continue

            # 对布尔型信号进行清洗
            if col.startswith(('playbook_', 'signal_', 'state_', 'event_', 'filter_')):
                # 检查并重命名王牌信号
                if col == 'signal_breakout_trigger_W':
                    new_col_name = 'BASE_SIGNAL_BREAKOUT_TRIGGER'
                    df_merged.rename(columns={col: new_col_name}, inplace=True)
                    df_merged[new_col_name] = df_merged[new_col_name].fillna(False).astype(bool)
                    # print(f"      - [清洗] 布尔信号 '{col}' -> '{new_col_name}' 已清洗 (NaN -> False)")
                else:
                    df_merged[col] = df_merged[col].fillna(False).astype(bool)
                    # print(f"      - [清洗] 布尔信号 '{col}' 已清洗 (NaN -> False)")
            
            # 对数值型分数进行清洗
            elif col.startswith(('washout_score_', 'rejection_signal_')):
                df_merged[col] = df_merged[col].fillna(0).astype(int)
                # print(f"      - [清洗] 数值分数 '{col}' 已清洗 (NaN -> 0)")
        
        # print("    - [数据流转] 全面NaN清洗完成。")
        
        return df_merged
    
    def _run_tactical_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """【引擎2】执行日线级别的战术分析和决策。"""
        # 调用日线策略实例的 apply_strategy 方法
        final_df, atomic_signals = self.tactical_engine.apply_strategy(
            all_dfs, self.tactical_config
        )

        if final_df is None or final_df.empty:
            return []
        
        # 准备数据库记录
        db_records = self.tactical_engine.prepare_db_records(
            stock_code, 
            final_df, 
            atomic_signals, 
            params=self.tactical_config,
            result_timeframe='D'
        )
        return db_records

    # ▼▼▼ 分钟共振引擎 ▼▼▼
    def _run_intraday_resonance_engine(self, stock_code: str, all_dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """【引擎3】执行分钟线级别的多周期共振检测。"""
        resonance_params = self.tactical_config.get('strategy_params', {}).get('trend_follow', {}).get('multi_level_resonance_params', {})
        if not resonance_params.get('enabled', False):
            logger.info("分钟共振剧本未启用，跳过引擎2。")
            return []
        levels = resonance_params.get('levels', [])
        if not levels:
            return []
        # 1. 确定触发周期（通常是最小周期）并准备数据
        trigger_level = levels[-1]
        trigger_tf = trigger_level['tf']
        if trigger_tf not in all_dfs or all_dfs[trigger_tf].empty:
            logger.warning(f"缺少触发周期 '{trigger_tf}' 的数据，无法运行分钟共振引擎。")
            return []
        # 2. 将所有周期的数据按时间对齐到触发周期上
        df_aligned = all_dfs[trigger_tf].copy()
        for level in levels[:-1]: # 遍历所有非触发周期
            level_tf = level['tf']
            if level_tf in all_dfs and not all_dfs[level_tf].empty:
                # 使用 merge_asof 将大周期数据广播到小周期上
                df_aligned = pd.merge_asof(
                    left=df_aligned,
                    right=all_dfs[level_tf],
                    left_index=True,
                    right_index=True,
                    direction='backward', # 关键：使用前一个已收盘的大周期K线状态
                    suffixes=('', f'_{level_tf}') # 为大周期列添加后缀以防冲突
                )
            else:
                logger.warning(f"共振检查缺少周期 '{level_tf}' 的数据，该级别的条件将无法满足。")
                return [] # 如果缺少关键数据，直接退出
        # 3. 逐级检查共振条件
        final_signal = pd.Series(True, index=df_aligned.index)
        for level in levels:
            level_tf = level['tf']
            level_logic = level.get('logic', 'AND').upper()
            level_conditions = level.get('conditions', [])
            level_signal = pd.Series(True if level_logic == 'AND' else False, index=df_aligned.index)
            for cond in level_conditions:
                # 调用辅助函数检查单个条件
                cond_signal = self._check_single_condition(df_aligned, cond, level_tf)
                if level_logic == 'AND':
                    level_signal &= cond_signal
                else: # OR
                    level_signal |= cond_signal
            final_signal &= level_signal
        # 4. 筛选出触发信号的行并生成记录
        triggered_df = df_aligned[final_signal]
        if triggered_df.empty:
            return []
        logger.info(f"分钟共振引擎在 {len(triggered_df)} 个时间点上发现共振信号。")
        db_records = []
        for timestamp, row in triggered_df.iterrows():
            record = self._prepare_intraday_db_record(stock_code, timestamp, row, resonance_params)
            db_records.append(record)
        return db_records

    def _check_single_condition(self, df: pd.DataFrame, cond: Dict, tf: str) -> pd.Series:
        """辅助函数：检查单个共振条件并返回布尔序列。"""
        cond_type = cond['type']
        trigger_tf = self.tactical_config.get('strategy_params', {}).get('trend_follow', {}).get('multi_level_resonance_params', {}).get('levels', [{}])[-1].get('tf')
        suffix = '' if tf == trigger_tf else f'_{tf}'
        if cond_type == 'ema_above':
            period = cond['period']
            ema_col = f'EMA_{period}{suffix}'
            close_col = f'close{suffix}'
            if ema_col in df.columns and close_col in df.columns:
                return df[close_col] > df[ema_col]
        elif cond_type == 'macd_above_zero':
            p = cond['periods']
            hist_col = f'MACDh_{p[0]}_{p[1]}_{p[2]}{suffix}'
            if hist_col in df.columns:
                return df[hist_col] > 0
        elif cond_type == 'macd_cross':
            p = cond['periods']
            hist_col = f'MACDh_{p[0]}_{p[1]}_{p[2]}{suffix}'
            if hist_col in df.columns:
                return (df[hist_col] > 0) & (df[hist_col].shift(1) <= 0)
        elif cond_type == 'dmi_cross':
            p = cond['period']
            pdi_col = f'DMP_{p}{suffix}'
            mdi_col = f'DMN_{p}{suffix}'
            if pdi_col in df.columns and mdi_col in df.columns:
                return (df[pdi_col] > df[mdi_col]) & (df[pdi_col].shift(1) <= df[mdi_col].shift(1))
        elif cond_type == 'kdj_cross':
            p = cond['periods']
            k_col, d_col = f'KDJk_{p[0]}_{p[1]}_{p[2]}{suffix}', f'KDJd_{p[0]}_{p[1]}_{p[2]}{suffix}'
            low_level = cond.get('low_level', 0)
            if k_col in df.columns and d_col in df.columns:
                return (df[k_col] > df[d_col]) & (df[k_col].shift(1) <= df[d_col].shift(1)) & (df[k_col] < low_level)
        elif cond_type == 'rsi_reversal':
            p = cond['period']
            rsi_col = f'RSI_{p}{suffix}'
            oversold_level = cond.get('oversold_level', 30)
            if rsi_col in df.columns:
                return (df[rsi_col] > oversold_level) & (df[rsi_col].shift(1) <= oversold_level)
        # 如果条件类型未知或缺少列，返回False
        return pd.Series(False, index=df.index)

    def _prepare_intraday_db_record(self, stock_code: str, timestamp: pd.Timestamp, row: pd.Series, params: dict) -> Dict[str, Any]:
        """辅助函数：为分钟线共振信号生成标准化的数据库记录。"""
        signal_name = params.get('signal_name', 'UNKNOWN_RESONANCE')
        trigger_tf = params['levels'][-1]['tf']
        record = {
            "stock_code": stock_code,
            "trade_time": sanitize_for_json(timestamp),
            "timeframe": trigger_tf, # 关键：使用触发周期的分钟级别
            "strategy_name": signal_name, # 使用共振信号的专属名称
            "close_price": sanitize_for_json(row.get('close')), # 使用触发周期的收盘价
            "entry_score": sanitize_for_json(params.get('score', 0.0)),
            "entry_signal": True, # 共振信号直接视为买入信号
            "exit_signal_code": 0,
            "is_long_term_bullish": False, # 分钟信号不直接判断长短期趋势
            "is_mid_term_bullish": False,
            "is_pullback_setup": False,
            "pullback_target_price": None,
            "triggered_playbooks": [signal_name],
            "context_snapshot": sanitize_for_json({'close': row.get('close')}),
        }
        return record
