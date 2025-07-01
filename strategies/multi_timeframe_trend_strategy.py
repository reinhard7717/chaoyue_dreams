# 文件: strategies/multi_timeframe_trend_strategy.py
# 版本: V3.0 - 分钟线监测修复与流程重构版
import asyncio
import logging
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from services.indicator_services import IndicatorService
from strategies.trend_following_strategy import TrendFollowStrategy
from strategies.weekly_trend_follow_strategy import WeeklyTrendFollowStrategy
from utils.config_loader import load_strategy_config

logger = logging.getLogger(__name__)

class MultiTimeframeTrendStrategy:

    def __init__(self):
        """
        【V3.0 分钟线监测修复与流程重构版】
        - 新增: 智能解析配置文件，自动识别所有需要的时间框架 (周、日、分钟)。
        - 重构: 统一数据准备流程，确保所有时间框架的数据被一次性、明确地请求和加载。
        - 修复: 解决了分钟线监测逻辑因数据缺失而失效的根本问题。
        """
        # 1. 加载所有需要的配置文件
        self.tactical_config_path = 'config/trend_follow_strategy.json'
        self.strategic_config_path = 'config/weekly_trend_follow_strategy.json'
        self.tactical_config = load_strategy_config(self.tactical_config_path)
        self.strategic_config = load_strategy_config(self.strategic_config_path)

        # 2. 实例化所有需要的服务和子策略
        self.indicator_service = IndicatorService()
        self.weekly_strategy = WeeklyTrendFollowStrategy()
        self.tactical_strategy = TrendFollowStrategy(config=self.tactical_config)

        # ▼▼▼【代码修改】: 新增智能解析逻辑，识别所有需要的时间框架 ▼▼▼
        self.required_timeframes = self._get_all_required_timeframes()
        print(f"--- [总指挥 MultiTimeframeTrendStrategy (V3.0)] 初始化完成 ---")
        print(f"    - [总指挥] 已智能识别出策略依赖的所有时间框架: {sorted(list(self.required_timeframes))}")
        # ▲▲▲【代码修改】: 修改结束 ▲▲▲

    # ▼▼▼【代码修改】: 新增辅助方法，用于解析配置 ▼▼▼
    def _get_all_required_timeframes(self) -> Set[str]:
        """
        智能解析战术和战略配置文件，找出所有需要加载数据的时间框架。
        """
        timeframes = {'W', 'D'} # 战略和战术核心，默认必须有

        # 解析战术配置中的分钟线需求
        if self.tactical_config:
            # 1. 从 vwap_confirmation_params 获取
            vwap_params = self.tactical_config.get('strategy_params', {}).get('trend_follow', {}).get('vwap_confirmation_params', {})
            if vwap_params.get('enabled', False):
                timeframes.add(vwap_params.get('timeframe', '5'))

            # 2. 从 multi_level_resonance_params 获取
            resonance_params = self.tactical_config.get('strategy_params', {}).get('trend_follow', {}).get('multi_level_resonance_params', {})
            if resonance_params.get('enabled', False):
                levels = resonance_params.get('levels', [])
                for level in levels:
                    if 'tf' in level:
                        timeframes.add(level['tf'])
        
        # 移除空值或None
        return {tf for tf in timeframes if tf}
    # ▲▲▲【代码修改】: 修改结束 ▲▲▲

    async def run_for_stock(self, stock_code: str, trade_time: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        【V3.0 流程重构版】为单个股票执行完整的多时间框架分析。
        - 核心修正: 使用统一的数据准备流程，确保所有识别出的时间框架数据都被加载。
        """
        logger.info(f"--- 开始为【{stock_code}】执行多时间框架分析 (V3.0 重构版) ---")

        # --- 步骤 1: 【重构】统一准备所有需要的数据 ---
        logger.info(f"--- 步骤1: 调用 IndicatorService 统一准备所有数据... ---")
        # 注意：这里我们假设 IndicatorService 的 prepare_data_for_strategy 能够根据
        # 配置文件中的 apply_on 字段，一次性准备好所有相关的时间框架数据。
        # 我们传递 tactical_config 因为它通常包含了所有指标的定义。
        all_dfs = await self.indicator_service.prepare_data_for_strategy(
            stock_code=stock_code, config=self.tactical_config, trade_time=trade_time
        )

        # 验证核心数据是否存在
        if not all_dfs or 'D' not in all_dfs or 'W' not in all_dfs:
            logger.warning(f"[{stock_code}] 核心数据(周线或日线)准备失败，分析终止。")
            return None
        
        # 验证所有必需的分钟线数据是否都已加载
        for tf in self.required_timeframes:
            if tf not in all_dfs:
                logger.warning(f"[{stock_code}] 警告：策略需要的分钟线周期 '{tf}' 未能成功加载，分钟线协同分析可能不完整。")

        df_weekly = all_dfs['W']
        df_daily = all_dfs['D']

        # --- 步骤 2: 运行战略层策略 (在纯周线数据上) ---
        logger.info(f"--- 步骤2: 运行周线战略策略，生成'战略信号'... ---")
        weekly_signals_df = self.weekly_strategy.apply_strategy(df_weekly)

        if weekly_signals_df is None or weekly_signals_df.empty:
            logger.warning(f"[{stock_code}] 周线战略策略未能生成战略背景，但将继续进行战术分析。")
            weekly_signals_df = pd.DataFrame(index=df_weekly.index)

        # --- 步骤 3: 将战略信号整合到日线数据中 ---
        logger.info(f"--- 步骤3: 整合战略信号到日线数据... ---")
        df_daily.index = pd.to_datetime(df_daily.index).tz_localize(None)
        weekly_signals_df.index = pd.to_datetime(weekly_signals_df.index).tz_localize(None)
        
        # 使用 all_dfs['D'] 来确保我们修改的是字典中的同一个对象
        all_dfs['D'] = pd.merge_asof(
            left=df_daily.sort_index(),
            right=weekly_signals_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )
        
        # 重命名和填充逻辑保持不变
        if 'signal_breakout_trigger_W' in all_dfs['D'].columns:
            all_dfs['D'].rename(columns={'signal_breakout_trigger_W': 'BASE_SIGNAL_BREAKOUT_TRIGGER'}, inplace=True)
            print("    - [协同层] 已将周线王牌信号 'signal_breakout_trigger_W' 重命名为 'BASE_SIGNAL_BREAKOUT_TRIGGER'")
        
        signal_cols = list(weekly_signals_df.columns)
        if 'BASE_SIGNAL_BREAKOUT_TRIGGER' in all_dfs['D'].columns:
            signal_cols.append('BASE_SIGNAL_BREAKOUT_TRIGGER')
            
        for col in signal_cols:
            if col in all_dfs['D'].columns:
                if all_dfs['D'][col].dtype == 'bool':
                    all_dfs['D'][col] = all_dfs['D'][col].fillna(False)
                else:
                    all_dfs['D'][col] = all_dfs['D'][col].fillna(0)
        
        print(f"    - [协同层] 已将 {len(weekly_signals_df.columns)} 个周线策略信号合并到日线。")
        print(f"    - [协同层] 最终数据集包含的周期: {list(all_dfs.keys())}")

        # --- 步骤 4: 运行战术层策略 (传入包含所有数据的字典) ---
        logger.info(f"--- 步骤4: 运行多时间框架战术策略... ---")
        final_df, atomic_signals = self.tactical_strategy.apply_strategy(
            all_dfs, self.tactical_config
        )

        # --- 步骤 5: 打包最终结果并返回 ---
        if final_df is None or final_df.empty:
            logger.info(f"\n--- 【{stock_code}】战术策略运行未产生有效结果DataFrame ---")
            return None
        
        logger.info(f"[{stock_code}] 战术策略分析完成，准备数据库记录...")
        db_records = self.tactical_strategy.prepare_db_records(
            stock_code, 
            final_df, 
            atomic_signals, 
            params=self.tactical_config
        )
        
        logger.info(f"--- 【{stock_code}】多时间框架分析完成，共生成 {len(db_records) if db_records else 0} 条信号记录。 ---")
        return db_records
