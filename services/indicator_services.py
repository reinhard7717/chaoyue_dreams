# services\indicator_services.py
import asyncio
import datetime
import json
from scipy.signal import find_peaks, peak_prominences
import traceback
import warnings
import logging
from dao_manager.tushare_daos.index_basic_dao import IndexBasicDAO
from dao_manager.tushare_daos.indicator_dao import IndicatorDAO
import numpy as np
import pandas as pd
from typing import List, Optional, Set, Union, Dict
import pandas_ta as ta
from dao_manager.tushare_daos.industry_dao import IndustryDao
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.index_basic_dao import IndexBasicDAO
from core.constants import TimeLevel
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO

warnings.filterwarnings(action='ignore', category=UserWarning, message='.*drop timezone information.*')
warnings.filterwarnings(action='ignore', category=FutureWarning, message=".*Passing 'suffixes' which cause duplicate columns.*")
pd.options.mode.chained_assignment = None

logger = logging.getLogger("services")

class IndicatorService:
    """
    技术指标计算服务 (使用 pandas-ta)
    负责获取多时间级别原始数据，进行时间序列标准化（重采样），计算指标，合并数据并进行最终填充。
    并行处理数据获取、重采样和指标计算任务以提高效率。
    """
    def __init__(self):
        """
        初始化 IndicatorService。
        设置 DAO 对象，并动态导入 pandas_ta 库。
        """
        self.indicator_dao = IndicatorDAO()
        self.industry_dao = IndustryDao()
        self.stock_basic_dao = StockBasicInfoDao()
        self.stock_trade_dao = StockTimeTradeDAO()
        self.index_dao = IndexBasicDAO()
        self.strategies_dao = StrategiesDAO() # 实例化DAO

        self.momentum_lookback = 60 # 动量计算回看周期
        self.fund_flow_lookback = 5   # 资金流计算回看周期
        
        try:
            global ta
            import pandas_ta as ta
            if ta is None:
                 logger.warning("pandas_ta 之前导入失败，尝试重新导入。")
                 import pandas_ta as ta
        except ImportError:
            logger.error("pandas-ta 库未安装，请运行 'pip install pandas-ta'")
            ta = None

    # ▼▼▼ 调试辅助函数，用于打印DataFrame时间范围 ▼▼▼
    def _log_df_time_range(self, df: Optional[pd.DataFrame], df_name: str):
        """【调试函数】打印DataFrame的时间范围、数据量等基本信息。"""
        print(f"    - [数据清查-范围] 正在检查 '{df_name}'...")
        if df is None or df.empty:
            print(f"      -> 结果: DataFrame为空或None。")
            return
        
        if not isinstance(df.index, pd.DatetimeIndex):
            print(f"      -> 结果: 索引不是 DatetimeIndex，无法分析时间范围。")
            return
            
        start_time = df.index.min().strftime('%Y-%m-%d %H:%M:%S')
        end_time = df.index.max().strftime('%Y-%m-%d %H:%M:%S')
        count = len(df)
        print(f"      -> 结果: 数据量={count}, 开始时间='{start_time}', 结束时间='{end_time}'")

    # ▼▼▼ 调试辅助函数，用于抽查数据对齐情况 ▼▼▼
    def _log_alignment_check(self, df: pd.DataFrame, num_samples: int = 10):
        """
        【调试函数 V2.0 - 全列展示版】
        随机抽查最终合并的DataFrame，检查所有列的数据对齐情况。
        """
        print(f"\n--- [数据清查-阶段2: 合并对齐抽查 (全列)] ---")
        if df.empty:
            print("    -> 抽查失败: DataFrame 为空。")
            print(f"--- [数据清查-阶段2: 检查完成] ---\n")
            return
        
        # 如果数据量小于抽样数，则展示所有数据
        if len(df) < num_samples:
            num_samples = len(df)
            print(f"    -> 数据量不足，展示全部 {num_samples} 行数据。")
        else:
            print(f"    -> 随机抽取 {num_samples} 个时间点，检查所有列的数据对齐情况:")

        # 使用固定的随机种子以保证每次抽查结果一致，便于调试
        sampled_df = df.sample(n=num_samples, random_state=42) if num_samples > 0 else df
        
        # 设置pandas显示选项，以确保所有列都能被打印出来，不会被省略
        with pd.option_context(
            'display.max_rows', None, 
            'display.max_columns', None, 
            'display.width', 200  # 调整宽度以适应控制台
        ):
            print(sampled_df)
            
        print(f"--- [数据清查-阶段2: 检查完成] ---\n")

    # ▼▼▼ 一个可复用的、健壮的时区标准化辅助函数 ▼▼▼
    def _standardize_df_index_to_utc(self, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        【V1.3 UTC源数据修复版】确保DataFrame的索引是UTC时区感知的。
        - 此版本根据“数据库源数据为UTC”进行修正。
        - 如果索引是“天真”的(naive, tz is None)，则直接将其本地化为UTC。
        - 如果索引已经是“感知”的(aware)，则统一转换为UTC以确保一致性。
        """
        if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return df
        # 创建副本以避免修改原始传入的DataFrame
        df_copy = df.copy()
        # ▼▼▼ 根据“源数据为UTC”的规则进行修正 ▼▼▼
        # 检查索引是否有时区信息
        if df_copy.index.tz is None:
            # 如果索引是“天真”的（naive），我们根据业务知识（数据库存的是UTC）
            # 直接使用 tz_localize 将其“标记”为 UTC 时区。
            # print(f"    - [时区标准化] 检测到 naive 时间索引，根据规则直接本地化为 'UTC'。")
            df_copy.index = df_copy.index.tz_localize('UTC')
        else:
            # 如果索引已经是“感知”的（aware），为保证统一，依然将其转换为 UTC
            # 这可以处理数据来源多样化，部分数据可能为其他时区的情况。
            # print(f"    - [时区标准化] 检测到 aware 时间索引，统一转换为 'UTC'。")
            df_copy.index = df_copy.index.tz_convert('UTC')
        return df_copy

    def _load_config(self, path: str) -> Dict:
        """
        【辅助函数】从给定的路径加载JSON配置文件。
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"    - 警告: 配置文件未找到: {path}")
            return {}
        except json.JSONDecodeError:
            print(f"    - 警告: 配置文件格式错误: {path}")
            return {}

    async def _get_ohlcv_data(self, stock_code: str, time_level: Union['TimeLevel', str], needed_bars: int, trade_time: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        【V2.0 最终版 - 核心数据获取函数】
        异步获取足够用于计算的原始历史数据 DataFrame。
        此函数仅负责从 DAO 获取并执行最通用的列名标准化，不进行任何特定于场景的数据准备。

        Args:
            stock_code (str): 股票代码。
            time_level (Union[TimeLevel, str]): 时间级别 ('D', 'W', '60', '30' 等)。
            needed_bars (int): 需要获取的 K 线数量。
            trade_time (Optional[str]): 交易时间，用于数据回溯。

        Returns:
            Optional[pd.DataFrame]: 包含原始 OHLCV 数据的 DataFrame，如果获取失败则为 None。
        """
        # print(f"    [底层数据获取] 正在为 {stock_code} 获取 {time_level} 级别数据，请求 {needed_bars} 条...")
        df = await self.indicator_dao.get_history_ohlcv_df(
            stock_code=stock_code, 
            time_level=time_level, 
            limit=needed_bars, 
            trade_time=trade_time
        )

        if df is None or df.empty:
            logger.warning(f"[{stock_code}] 时间级别 {time_level} 无法获取到数据。")
            return None

        # 通用的列名标准化
        if 'vol' in df.columns and 'volume' not in df.columns:
            df.rename(columns={'vol': 'volume'}, inplace=True)

        # 确保数据有 DatetimeIndex，这是后续所有时间序列操作的基础
        # 这是从 _fetch_and_prepare_base_data 吸收的优点，但放在这里是合理的，因为它是通用的准备步骤
        if not isinstance(df.index, pd.DatetimeIndex):
            time_col = None
            if 'trade_date' in df.columns: # 适用于日线、周线
                time_col = 'trade_date'
            elif 'trade_time' in df.columns: # 适用于分钟线
                time_col = 'trade_time'
            
            if time_col:
                df[time_col] = pd.to_datetime(df[time_col])
                df.set_index(time_col, inplace=True)
                # print(f"    [底层数据获取] 已将 '{time_col}' 列设置为 DatetimeIndex。")
            else:
                logger.error(f"[{stock_code}] {time_level} 数据既没有 DatetimeIndex，也没有 'trade_date'/'trade_time' 列。")
                return None

        logger.debug(f"[{stock_code}] 时间级别 {time_level} 获取到 {len(df)} 条原始K线数据。")
        return df

    def _find_params_recursively(self, config_dict: Dict, key_to_find: str) -> Optional[Dict]:
        """
        在配置字典中递归查找指定的键。

        Args:
            config_dict (Dict): 要搜索的配置字典。
            key_to_find (str): 要查找的键名 (例如 'industry_context_params')。

        Returns:
            Optional[Dict]: 如果找到，返回对应的子字典；否则返回 None。
        """
        if key_to_find in config_dict:
            return config_dict[key_to_find]
        
        for key, value in config_dict.items():
            if isinstance(value, dict):
                result = self._find_params_recursively(value, key_to_find)
                if result is not None:
                    return result
        return None

    def _discover_required_timeframes_from_config(self, config: Dict) -> Set[str]:
        """
        【V7.2 终极递归版】通过递归扫描整个配置，智能、全面地找出所有需要加载数据的时间框架。
        - 核心升级: 放弃了脆弱的、基于固定路径的扫描方式。
        - 解决方案: 实现了一个递归辅助函数，可以遍历配置字典的每一个角落，
                     查找所有名为 'apply_on', 'timeframe', 'tf' 的键，从而确保不会遗漏任何周期需求。
                     这是解决周期发现问题的根本性方法。
        """
        timeframes: Set[str] = set()

        def _recursive_search(data):
            """
            【辅助函数】递归遍历字典和列表，查找并收集所有时间框架。
            """
            if isinstance(data, dict):
                for key, value in data.items():
                    # 关键匹配逻辑
                    if key == 'apply_on' and isinstance(value, list):
                        for tf in value:
                            if tf: timeframes.add(str(tf))
                    elif key in ['timeframe', 'tf'] and isinstance(value, (str, int)):
                        if value: timeframes.add(str(value))
                    
                    # 继续向内层递归
                    if isinstance(value, (dict, list)):
                        _recursive_search(value)
            
            elif isinstance(data, list):
                for item in data:
                    # 继续向内层递归
                    if isinstance(item, (dict, list)):
                        _recursive_search(item)

        # 从配置的根节点开始进行全局递归搜索
        _recursive_search(config)

        # 安全保障：如果配置了任何指标，至少应包含日线数据，因为周/月线重采样依赖日线
        if config.get('feature_engineering_params', {}).get('indicators'):
            timeframes.add('D')

        # 移除可能存在的空字符串并返回
        return {tf for tf in timeframes if tf}

    async def prepare_data_for_strategy(
        self,
        stock_code: str,
        config: dict,
        trade_time: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        【V6.2 终极自适应版】为策略准备数据的统一入口。
        该方法会递归搜索配置，自适应地决定是否附加行业背景数据。

        Args:
            stock_code (str): 股票代码。
            config (dict): 任何策略的完整配置文件字典。
            trade_time (Optional[str]): 交易时间。

        Returns:
            Dict[str, pd.DataFrame]: 包含所有时间周期DataFrame的字典。
        """
        # print(f"--- [数据准备V6.2-自适应版] 开始为 {stock_code} 准备数据... ---")

        # --- 步骤 1: 始终执行基础数据和指标的准备 ---
        all_dfs = await self._prepare_base_data_and_indicators(stock_code, config, trade_time)

        if not all_dfs or 'D' not in all_dfs or all_dfs['D'].empty:
            logger.warning(f"[{stock_code}] 基础数据准备失败，无法继续。")
            return all_dfs

        # --- 步骤 2: 【核心升级】使用递归搜索来检查配置 ---
        industry_params = self._find_params_recursively(config, 'industry_context_params')
        is_industry_enabled = industry_params.get('enabled', False) if industry_params else False

        if not is_industry_enabled:
            print("    - [配置信息] 在策略配置中未找到启用的 'industry_context_params'，跳过行业强度计算。")
            # print(f"--- [数据准备V6.2-自适应版] {stock_code} 数据准备完成（无行业背景）。 ---")
            return all_dfs

        # --- 步骤 3: 如果启用，则执行行业数据注入 (此部分逻辑不变) ---
        print(f"    - [配置信息] 检测到行业协同已启用，开始获取行业强度...")
        
        current_trade_date = pd.to_datetime(trade_time).date() if trade_time else datetime.datetime.now().date()
        industry_rank_df = await self.calculate_industry_strength_rank(current_trade_date)
        stock_industry_info = await self.indicator_dao.get_stock_industry_info(stock_code)
        stock_industry_code = stock_industry_info.get('code') if stock_industry_info else None
        stock_industry_name = stock_industry_info.get('name') if stock_industry_info else '未知行业'

        stock_industry_rank = 0.0
        if not industry_rank_df.empty and stock_industry_code and stock_industry_code in industry_rank_df.index:
            stock_industry_rank = industry_rank_df.loc[stock_industry_code, 'strength_rank']
            print(f"    - [行业背景注入] 股票 {stock_code} 所属行业 '{stock_industry_name}'({stock_industry_code}) 当日强度排名: {stock_industry_rank:.2%}")
        else:
            print(f"    - [行业背景注入] 股票 {stock_code} ({stock_industry_name}) 未找到行业排名，默认排名为 0.0。")

        all_dfs['D']['industry_strength_rank_D'] = stock_industry_rank
        print(f"    - [行业背景注入] 已将 'industry_strength_rank_D' 列 ({stock_industry_rank:.2f}) 添加到日线数据。")
        
        # print(f"--- [数据准备V6.2-自适应版] {stock_code} 数据准备完成（含行业背景）。 ---")
        return all_dfs

    async def _prepare_base_data_and_indicators(
        self,
        stock_code: str,
        config: dict,
        trade_time: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        【V7.6 架构优化版】
        - 核心重构: 移除了独立获取CYQ数据的逻辑，统一由 StrategiesDAO.get_fund_flow_and_chips_data 提供所有补充数据。
                    这消除了数据获取的冗余，并从根本上解决了因不同处理方式（ffill vs fillna(0)）导致的数据冲突问题。
        - 流程简化: 现在只有一个补充数据源（df_supplemental），代码更清晰，责任更明确。
        - 性能提升: 调用DAO时传入limit参数，避免全量查询，提升效率。
        """
        # 1. 从配置中解析需要哪些时间周期的数据
        required_tfs = self._discover_required_timeframes_from_config(config)
        base_needed_bars = config.get('feature_engineering_params', {}).get('base_needed_bars', 500)
        print(f"    - [配置读取] 策略请求的基础数据量 (base_needed_bars) 为: {base_needed_bars}")
        
        indicators_config = config.get('feature_engineering_params', {}).get('indicators', {})
        
        # 2. 判断是否需要获取补充数据（资金流、筹码等）
        needs_supplemental_data = any(
            params.get('enabled', False) and key in [
                'advanced_fund_features', 'chip_cost_breakthrough', 
                'chip_pressure_release', 'winner_rate_reversal', 'capital_flow_divergence' # 扩展检查范围
            ]
            for key, params in indicators_config.items() if isinstance(params, dict)
        )

        if not required_tfs:
            return {}

        # 3. 准备所有数据获取任务
        base_tfs_to_fetch = set()
        resample_map = {}
        for tf in required_tfs:
            if tf in ['W', 'M']:
                base_tfs_to_fetch.add('D')
                resample_map[tf] = 'D'
            else:
                base_tfs_to_fetch.add(tf)

        tasks = []
        async def _fetch_and_tag_data(tf_to_fetch, bars_to_fetch, trade_time_str):
            df = await self._get_ohlcv_data(stock_code, tf_to_fetch, bars_to_fetch, trade_time_str)
            return (tf_to_fetch, df)

        for tf in base_tfs_to_fetch:
            bars_to_fetch = base_needed_bars
            if tf == 'D' and resample_map: # 如果需要周线或月线，日线数据需要更多
                bars_to_fetch = max(bars_to_fetch, 1000)
            tasks.append(_fetch_and_tag_data(tf, bars_to_fetch, trade_time))
        
        # ▼▼▼【核心架构】: 统一调用增强后的DAO方法获取所有补充数据 ▼▼▼
        if needs_supplemental_data:
            async def _fetch_supplemental_data_tagged(stock_code, trade_time, limit):
                trade_time_dt = pd.to_datetime(trade_time) if trade_time else None
                # 调用我们刚刚优化的DAO方法，并传入limit参数
                df = await self.strategies_dao.get_fund_flow_and_chips_data(stock_code, trade_time_dt, limit)
                return ('supplemental', df)
            # 将基础数据量作为limit传入，确保数据量匹配
            tasks.append(_fetch_supplemental_data_tagged(stock_code, trade_time, base_needed_bars))
        
        # 4. 并发执行所有数据获取任务
        all_data_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        raw_dfs: Dict[str, pd.DataFrame] = {}
        df_supplemental: Optional[pd.DataFrame] = None

        for result in all_data_results:
            if isinstance(result, Exception): continue
            if not (isinstance(result, tuple) and len(result) == 2): continue
            
            tag, data = result
            if tag == 'supplemental':
                if isinstance(data, pd.DataFrame):
                    df_supplemental = data
            else:
                tf = tag
                df = data
                if isinstance(df, pd.DataFrame) and not df.empty:
                    raw_dfs[tf] = df

        if 'D' not in raw_dfs:
            logger.error(f"[{stock_code}] 最核心的日线数据获取失败，处理终止。")
            return {}
        
        print(f"    - [数据流追踪] 步骤1: 原始日线数据已加载，行数: {len(raw_dfs['D'])}")
        
        # 5. 执行重采样（周线、月线）
        if 'D' in raw_dfs and resample_map:
            df_daily = raw_dfs['D']
            for target_tf, source_tf in resample_map.items():
                if source_tf == 'D' and not df_daily.empty:
                    ohlc_rule = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
                    resample_period = 'W-FRI' if target_tf == 'W' else 'M'
                    df_resampled = df_daily.resample(resample_period).agg(ohlc_rule)
                    df_resampled.dropna(inplace=True)
                    if not df_resampled.empty:
                        raw_dfs[target_tf] = df_resampled

        # 6. 标准化索引并计算指标
        for tf, df in raw_dfs.items():
            if df is not None and not df.empty:
                raw_dfs[tf] = self._standardize_df_index_to_utc(df)

        processed_dfs: Dict[str, pd.DataFrame] = {}
        calc_tasks = []

        async def _calculate_for_tf(tf, df):
            if tf == 'D':
                # ▼▼▼【核心架构】: 简化数据融合逻辑，只处理一个来源，并使用正确的ffill ▼▼▼
                if df_supplemental is not None and not df_supplemental.empty:
                    print("    - [数据融合] 正在将补充数据(资金流、筹码等)合并到日线...")
                    df_supplemental_std = self._standardize_df_index_to_utc(df_supplemental)
                    print(f"    - [数据流追踪] 步骤2: 日线合并前，行数: {len(df)}, 补充数据行数: {len(df_supplemental_std)}")
                    df = pd.merge(df, df_supplemental_std, left_index=True, right_index=True, how='left')
                    df[list(df_supplemental_std.columns)] = df[list(df_supplemental_std.columns)].ffill()
                    print(f"    - [数据流追踪] 步骤3: 日线合并后，行数: {len(df)}, 列: {df.columns.tolist()}")
            
            df_with_indicators = await self._calculate_indicators_for_timescale(df, indicators_config, tf)

            if tf == 'D':
                print(f"    - [数据流追踪] 步骤4: 日线指标计算后，行数: {len(df_with_indicators)}, 列: {df_with_indicators.columns.tolist()}")
                self._log_alignment_check(df_with_indicators)
            
            return tf, df_with_indicators

        for tf, df in raw_dfs.items():
            if tf in required_tfs:
                calc_tasks.append(_calculate_for_tf(tf, df))

        processed_results = await asyncio.gather(*calc_tasks, return_exceptions=True)

        for res in processed_results:
            if isinstance(res, Exception):
                logger.error(f"指标计算任务中出现异常: {res}", exc_info=True)
                continue
            if res and isinstance(res, tuple) and len(res) == 2:
                tf, df_processed = res
                if df_processed is not None and not df_processed.empty:
                    processed_dfs[tf] = df_processed
                else:
                    logger.warning(f"周期 '{tf}' 的指标计算结果为空DataFrame，已被丢弃。")

        print(f"--- [数据准备V7.7-日志增强版] 数据准备完成，最终字典包含的周期: {sorted(list(processed_dfs.keys()))} ---")
        return processed_dfs

    def _get_max_period_for_timeframe(self, config: dict, timeframe_key: str) -> int:
        """
        解析指标配置，获取指定时间周期所需的最大计算周期。
        """
        max_period = 0
        for indicator_key, params in config.items():
            if not params.get('enabled', False) or timeframe_key not in params.get("apply_on", []):
                continue
            configs_to_process = params.get('configs', [params])
            for sub_config in configs_to_process:
                periods = sub_config.get('periods')
                if periods is None: continue
                flat_periods = []
                if isinstance(periods, list):
                    for p in periods:
                        if isinstance(p, list): flat_periods.extend(p)
                        elif isinstance(p, (int, float)): flat_periods.append(p)
                elif isinstance(periods, (int, float)): flat_periods.append(periods)
                if flat_periods: max_period = max(max_period, max(flat_periods))
        return int(max_period * 1.2) if max_period > 0 else 1

    async def _calculate_indicators_for_timescale(self, df: pd.DataFrame, config: dict, timeframe_key: str) -> pd.DataFrame:
        """
        【V5.8 终极合并修复版】根据配置为指定时间周期计算所有技术指标。
        - 核心修复: 修正了V5.7版本中错误的合并逻辑，确保基础OHLCV列（如 high, close）也能被正确添加后缀（如 high_W, close_W），彻底解决下游策略的 'KeyError' 问题。
        - 保持健壮: 保留了V5.7引入的对Series返回值的健壮处理和数据量预检功能。
        """
        # print(f"  [指标计算V5.8] 开始为周期 '{timeframe_key}' 计算指标...")
        if not config:
            print(f"    - 警告: 周期 '{timeframe_key}' 没有配置任何指标。")
            return df

        max_required_period = self._get_max_period_for_timeframe(config, timeframe_key)
        if len(df) < max_required_period:
            logger.warning(f"数据行数 ({len(df)}) 不足以满足周期 '{timeframe_key}' 的最大计算要求 ({max_required_period})，将跳过该周期的所有指标计算。")
            return df

        df_main = df.copy()
        # base_cols = ['open', 'high', 'low', 'close', 'volume']
        # df_for_calc = pd.DataFrame(index=df_main.index)
        # for col in base_cols:
        #     if col in df_main.columns:
        #         df_for_calc[col] = df_main[col]
        #     else:
        #         df_for_calc[col] = np.nan 

        df_for_calc = df_main.copy()
        
        indicator_method_map = {
            'ema': self.calculate_ema, 'vol_ma': self.calculate_vol_ma, 'trix': self.calculate_trix,
            'coppock': self.calculate_coppock, 'rsi': self.calculate_rsi, 'macd': self.calculate_macd,
            'dmi': self.calculate_dmi, 'roc': self.calculate_roc, 'boll_bands_and_width': self.calculate_boll_bands_and_width,
            'cmf': self.calculate_cmf, 'bias': self.calculate_bias, 'atrn': self.calculate_atrn,
            'atrr': self.calculate_atrr, 'obv': self.calculate_obv, 'kdj': self.calculate_kdj,
            'uo': self.calculate_uo, 'vwap': self.calculate_vwap,
            'consolidation_period': self.calculate_consolidation_period,
            'advanced_fund_features': self.calculate_advanced_fund_features,
            'fibonacci_levels': self.calculate_fibonacci_levels,
        }
        
        def merge_results(result_data, target_df):
            """健壮地合并结果，无论输入是Series还是DataFrame。"""
            if result_data is None or result_data.empty: return
            if isinstance(result_data, pd.Series):
                result_data = result_data.to_frame()
            
            if isinstance(result_data, pd.DataFrame):
                for col in result_data.columns:
                    target_df[col] = result_data[col]
            else:
                logger.warning(f"指标计算返回了未知类型 {type(result_data)}，已跳过。")

        # --- 常规指标计算循环 (逻辑不变) ---
        for indicator_key, params in config.items():
            indicator_name = indicator_key.lower()
            
            if indicator_name in ['说明', 'index_sync', 'cyq_perf', 'zscore'] or not params.get('enabled', False): continue
            if indicator_name not in indicator_method_map:
                logger.warning(f"    - 警告: 未找到指标 '{indicator_name}' 的计算方法，已跳过。")
                continue
            if indicator_name in ['consolidation_period', 'advanced_fund_features', 'fibonacci_levels']: continue

            configs_to_process = params.get('configs', [params])
            for sub_config in configs_to_process:
                if timeframe_key not in sub_config.get("apply_on", []): continue
                try:
                    method_to_call = indicator_method_map[indicator_name]
                    kwargs = {'df': df_for_calc}
                    periods = sub_config.get('periods')

                    if indicator_name == 'vwap':
                        anchor = 'D' if timeframe_key.isdigit() else timeframe_key
                        kwargs['anchor'] = anchor
                        result_df = await method_to_call(**kwargs)
                        merge_results(result_df, df_for_calc)
                        continue

                    if periods is None:
                        result_df = await method_to_call(**kwargs)
                        merge_results(result_df, df_for_calc)
                        continue
                    
                    is_multi_param = indicator_name in ['macd', 'trix', 'coppock', 'kdj', 'uo']
                    is_nested_list = isinstance(periods[0], list) if periods else False
                    periods_to_iterate = [periods] if is_multi_param and not is_nested_list else periods

                    for p_set in periods_to_iterate:
                        kwargs_iter = {'df': df_for_calc}
                        if indicator_name == 'macd': kwargs_iter.update({'period_fast': p_set[0], 'period_slow': p_set[1], 'signal_period': p_set[2]})
                        elif indicator_name == 'trix': kwargs_iter.update({'period': p_set[0], 'signal_period': p_set[1]})
                        elif indicator_name == 'coppock': kwargs_iter.update({'long_roc_period': p_set[0], 'short_roc_period': p_set[1], 'wma_period': p_set[2]})
                        elif indicator_name == 'kdj': kwargs_iter.update({'period': p_set[0], 'signal_period': p_set[1], 'smooth_k_period': p_set[2]})
                        elif indicator_name == 'uo': kwargs_iter.update({'short_period': p_set[0], 'medium_period': p_set[1], 'long_period': p_set[2]})
                        elif indicator_name == 'boll_bands_and_width':
                            kwargs_iter.update({'period': p_set, 'std_dev': float(sub_config.get('std_dev', 2.0))})
                        else:
                            # 修正: 确保传递给 'period' 的是整数。如果 p_set 是列表(如[13])，则取其第一个元素；如果已经是整数(如13)，则直接使用。
                            kwargs_iter['period'] = p_set[0] if isinstance(p_set, list) else p_set
                        
                        result_df = await method_to_call(**kwargs_iter)
                        merge_results(result_df, df_for_calc)
                except Exception as e:
                    logger.error(f"    - 计算指标 {indicator_name.upper()} (周期: {timeframe_key}, 参数: {sub_config.get('periods')}) 时出错: {e}", exc_info=True)

        # --- 复合指标计算循环 (逻辑不变) ---
        for indicator_key, params in config.items():
            indicator_name = indicator_key.lower()
            if indicator_name in ['consolidation_period', 'advanced_fund_features', 'fibonacci_levels'] and params.get('enabled', False):
                if timeframe_key in params.get("apply_on", []):
                    try:
                        method_to_call = indicator_method_map[indicator_name]
                        result_df = await method_to_call(df=df_for_calc, params=params, suffix='')
                        merge_results(result_df, df_for_calc)
                    except Exception as e:
                        logger.error(f"    - 复合指标 {indicator_name.upper()} (周期: {timeframe_key}) 计算时出错: {e}", exc_info=True)

        # ▼▼▼ 后处理阶段：计算Z-Score ▼▼▼
        # 解释：Z-Score依赖于其他指标（如MACD）的计算结果，因此必须在所有常规指标计算完毕后执行。
        zscore_params = config.get('zscore')
        if zscore_params and zscore_params.get('enabled', False):
            for z_config in zscore_params.get('configs', []):
                if timeframe_key not in z_config.get("apply_on", []):
                    continue

                try:
                    source_pattern = z_config.get("source_column_pattern")
                    output_col_name_final = z_config.get("output_column_name") # 这是最终想要的列名，如 MACD_HIST_ZSCORE_D
                    window = z_config.get("window", 60)

                    if not all([source_pattern, output_col_name_final, window]):
                        logger.warning(f"Z-score配置不完整，跳过: {z_config}")
                        continue

                    # 动态构建带后缀的源列名
                    source_col_name_final = source_pattern
                    if "{fast}" in source_pattern:
                        macd_cfg = config.get('macd', {})
                        macd_periods = next((c.get('periods') for c in macd_cfg.get('configs', []) if timeframe_key in c.get('apply_on', [])), None)
                        if macd_periods and len(macd_periods) == 3:
                            source_col_name_final = source_pattern.format(fast=macd_periods[0], slow=macd_periods[1], signal=macd_periods[2])
                        else:
                            logger.warning(f"无法为Z-score找到周期为'{timeframe_key}'的MACD参数，跳过。")
                            continue
                    
                    # 【核心修复】根据当前周期，移除列名中的后缀，得到在 df_for_calc 中实际使用的内部列名
                    internal_suffix = f"_{timeframe_key}"
                    internal_source_col = source_col_name_final.removesuffix(internal_suffix) if source_col_name_final.endswith(internal_suffix) else source_col_name_final
                    internal_output_col = output_col_name_final.removesuffix(internal_suffix) if output_col_name_final.endswith(internal_suffix) else output_col_name_final

                    # 检查内部源列是否存在于 df_for_calc
                    if internal_source_col not in df_for_calc.columns:
                        logger.warning(f"Z-score计算失败：内部源列 '{internal_source_col}' 在临时DataFrame中不存在。请检查源指标是否已成功计算。")
                        continue

                    # 执行Z-score计算
                    source_series = df_for_calc[internal_source_col]
                    rolling_mean = source_series.rolling(window=window, min_periods=1).mean()
                    rolling_std = source_series.rolling(window=window, min_periods=1).std()
                    
                    # 使用 np.divide 安全地处理除以0的情况
                    zscore_result = np.divide(
                        (source_series - rolling_mean), 
                        rolling_std, 
                        out=np.full_like(source_series, np.nan), 
                        where=rolling_std!=0
                    )
                    
                    # 将结果写入到内部DataFrame，使用不带后缀的列名
                    df_for_calc[internal_output_col] = zscore_result
                    
                    print(f"    - [指标计算-ZScore] 已成功为列 '{internal_source_col}' 计算Z-score，临时输出到 '{internal_output_col}'。")

                except Exception as e:
                    logger.error(f"计算Z-score时出错 (配置: {z_config}): {e}", exc_info=True)

        # 1. 确定后缀。日线、周线、月线加后缀，分钟线等保持原样。
        suffix = f"_{timeframe_key}" if timeframe_key in ['D', 'W', 'M'] else ''
        
        # 如果不需要加后缀（例如分钟线），直接返回包含所有计算结果的 df_for_calc
        if not suffix:
            return df_for_calc

        # 2. 对于需要加后缀的周期（D, W, M）:
        # 创建一个重命名的映射字典，为 df_for_calc 中的每一列都规划好带后缀的新名字。
        # 这样可以确保数据的一致性，所有列都会被统一处理。
        rename_map = {col: f"{col}{suffix}" for col in df_for_calc.columns}
        
        # 3. 应用重命名，生成最终的DataFrame
        final_df = df_for_calc.rename(columns=rename_map)

        # 4. 调试输出，检查最终返回的数据是否符合预期
        debug_cols_to_check = [
            f'close{suffix}', f'winner_rate{suffix}', f'weight_avg{suffix}', 
            f'cost_85pct{suffix}', f'cost_95pct{suffix}'
        ]
        # 筛选出实际存在于final_df中的列进行打印，避免KeyError
        existing_debug_cols = [col for col in debug_cols_to_check if col in final_df.columns]

        if existing_debug_cols:
            debug_df = final_df[(final_df.index >= '2024-11-01') & (final_df.index <= '2024-12-31')]
            if not debug_df.empty:
                print(f"\n--- [IndicatorService V6.0 最终输出 - 周期 {timeframe_key}] ---")
                with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 220):
                    print(debug_df[existing_debug_cols])
                print(f"--- [IndicatorService V6.0 最终输出结束 - 周期 {timeframe_key}] ---\n")

        return final_df

    async def calculate_industry_strength_rank(self, trade_date: datetime.date, market_code: str = '000905.SH') -> pd.DataFrame:
        """
        【V2.0 结构分析版】计算指定交易日所有行业的强度分及排名。
        综合了价量趋势、资金流向、龙头效应、板块协同性和涨停梯队。
        """
        # print(f"--- [IndustryService V2.1] 开始计算 {trade_date} 的行业结构化强度 (对比基准: {market_code}) ---")
        start_date = trade_date - datetime.timedelta(days=self.momentum_lookback + 30)
        market_daily_df = await self.indicator_dao.get_market_index_daily_data(market_code, start_date, trade_date)
        if market_daily_df.empty:
            print(f"    - 严重警告: 无法获取大盘基准 {market_code} 数据，相对强度分析将跳过。")
        
        # 1. 获取所有行业列表
        all_industries = await self.indicator_dao.get_all_industries()
        if not all_industries:
            print("    - 警告: 未找到任何行业，计算中止。")
            return pd.DataFrame()

        strength_data = []
        # 使用 asyncio.gather 并行处理所有行业
        tasks = [self._process_single_industry_strength(industry, trade_date, market_daily_df) for industry in all_industries]
        results = await asyncio.gather(*tasks)
        
        # 过滤掉计算失败的结果
        strength_data = [res for res in results if res is not None]

        if not strength_data:
            print("--- [IndustryService V2.0] 计算完成，未找到有效的行业数据。 ---")
            return pd.DataFrame()

        # 5. 归一化排名
        df = pd.DataFrame(strength_data)
        # strength_score 已经是0-100分，可以直接用。rank(pct=True)是相对排名。
        df['strength_rank'] = df['strength_score'].rank(pct=True, ascending=True) 
        
        print(f"--- [IndustryService V2.0] {trade_date} 的行业结构化强度计算完成。 ---")
        return df.sort_values('strength_rank', ascending=False).set_index('industry_code')

    async def _process_single_industry_strength(self, industry, trade_date: datetime.date, market_daily_df: pd.DataFrame) -> Optional[Dict]:
        """
        【新增】处理单个行业的强度计算，便于并行化。
        """
        # print(f"  - 正在处理行业: {industry.name} ({industry.ts_code})")

        # 2. 并行获取该行业所需的所有数据
        start_date = trade_date - datetime.timedelta(days=self.momentum_lookback + 30)

        try:
            # 使用 asyncio.gather 获取一个行业的所有数据
            data_tasks = {
                "daily": self.indicator_dao.get_industry_daily_data(industry.ts_code, start_date, trade_date),
                "flow": self.indicator_dao.get_industry_fund_flow(industry.ts_code, start_date, trade_date)
            }
            data_results = await asyncio.gather(*data_tasks.values())
            
            industry_daily_df = data_results[0]
            industry_fund_flow_df = data_results[1]

            if industry_daily_df.empty:
                # print(f"    - 跳过 {industry.name}: 无有效的日线行情数据。")
                return None

            # 3. 计算各项基础得分 (0-1分制)
            momentum_score = self._calculate_momentum_score(industry_daily_df, trade_date) # 沿用旧方法
            fund_flow_score = self._calculate_fund_flow_score(industry_fund_flow_df, trade_date) # 沿用旧方法
            volume_score = await self._calculate_volume_profile_score(industry_daily_df)
            rs_score = await self._calculate_relative_strength_score(industry_daily_df, market_daily_df)

            # 4. 【核心升级】计算结构化得分 (0-1分制)
            leader_score = await self._calculate_leader_score(industry.ts_code, trade_date)
            cohesion_score = await self._calculate_cohesion_score(industry.ts_code, trade_date)
            echelon_score = await self._calculate_limit_up_echelon_score(industry.ts_code, trade_date)

            # 5. 加权合成总分 (总分100分)
            # 权重分配：结构 > 资金 > 趋势
            # 涨停梯队是最高优先级信号，给予一票否决权的权重
            total_score = (
                10 * momentum_score +          # 趋势基础分
                15 * fund_flow_score +         # 资金跟随分
                10 * volume_score +            # 成交活跃分
                15 * rs_score +                # 【新增】相对强度分
                15 * leader_score +            # 龙头效应分
                15 * cohesion_score +          # 板块协同分
                30 * echelon_score             # 【核心】涨停梯队分
            )
            # echelon_score 的权重可以更高，比如40，其他相应降低，以体现其重要性

            return {
                'industry_code': industry.ts_code,
                'industry_name': industry.name,
                'strength_score': total_score
            }
        except Exception as e:
            print(f"    - 处理行业 {industry.name} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def _calculate_volume_profile_score(self, industry_daily_df: pd.DataFrame) -> float:
        """
        【V1.1 健壮性修正版】计算行业成交活跃度得分。
        - 修正了在数据量不足时 rolling 操作返回 NaN 的问题。
        - 增加了对计算结果的 NaN 值检查，避免程序异常。
        """
        # ▼▼▼ 增加数据长度检查，如果数据太少则直接返回0分 ▼▼▼
        # 解释: 至少需要5天数据才能计算5日均线，否则分析无意义。
        if industry_daily_df.empty or 'turnover_rate' not in industry_daily_df.columns or len(industry_daily_df) < 5:
            # print(f"      - [成交活跃度] 数据不足 (行数: {len(industry_daily_df)})，无法计算得分。")
            return 0.0

        df = industry_daily_df.copy()

        # 1. 计算短期爆发强度：当日换手率在过去60个交易日中的百分位排名
        # 解释: 设置 min_periods=20 确保即使数据不足60天，只要超过20天也能计算出排名。
        turnover_rank_60d = df['turnover_rate'].rolling(60, min_periods=20).rank(pct=True).iloc[-1]
        
        # 2. 计算中期趋势：5日均线是否上穿20日均线
        # 解释: 设置 min_periods=1 确保均线计算从一开始就有值。
        df['turnover_ma5'] = df['turnover_rate'].rolling(5, min_periods=1).mean()
        df['turnover_ma20'] = df['turnover_rate'].rolling(20, min_periods=1).mean()
        
        # 判断金叉发生在最近3天内
        was_below = df['turnover_ma5'].shift(1) < df['turnover_ma20'].shift(1)
        is_above = df['turnover_ma5'] > df['turnover_ma20']
        is_cross_today = was_below & is_above
        is_recent_cross_series = is_cross_today.rolling(3, min_periods=1).sum()
        is_recent_cross = is_recent_cross_series.iloc[-1] > 0 if not is_recent_cross_series.empty else False

        # 3. 综合评分
        score = 0.0
        # ▼▼▼ 增加对 turnover_rank_60d 的 NaN 检查 ▼▼▼
        # 解释: 在进行比较和评分前，必须确保值不是 NaN。
        if pd.notna(turnover_rank_60d):
            # 如果短期爆发力极强 (排名前10%)，给予高分
            if turnover_rank_60d > 0.9:
                score += 0.6
        
        # 如果中期趋势向好 (近期金叉)，给予加分
        if is_recent_cross:
            score += 0.4
            
        # ▼▼▼ 格式化输出前也进行 NaN 检查，使日志更清晰 ▼▼▼
        # rank_str = f"{turnover_rank_60d:.2%}" if pd.notna(turnover_rank_60d) else "N/A"
        # print(f"      - [成交活跃度] 当日换手率排名: {rank_str}, 近期均线金叉: {is_recent_cross}, 得分: {score:.2f}")
        
        return score

    # --- 所有指标计算函数 async def calculate_* ---
    async def calculate_atr(self, df: pd.DataFrame, period: int = 14, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算 ATR (平均真实波幅)"""
        required_cols = [high_col, low_col, close_col]
        if df is None or df.empty or not all(col in df.columns for col in required_cols):
            logger.warning(f"计算 ATR 缺少必要列: {required_cols}。可用列: {df.columns.tolist() if df is not None else 'None'}")
            return None
        if len(df) < period:
            logger.warning(f"数据行数 ({len(df)}) 不足以计算周期为 {period} 的 ATR。")
            return None
        try:
            def _sync_atr():
                return ta.atr(high=df[high_col], low=df[low_col], close=df[close_col], length=period)
            atr_series = await asyncio.to_thread(_sync_atr)
            if atr_series is None or atr_series.empty:
                logger.warning(f"ATR_{period} 计算结果为空。")
                return None
            df_results = pd.DataFrame({f'ATR_{period}': atr_series}, index=df.index)
            return df_results
        except Exception as e:
            logger.error(f"计算 ATR (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_atrr(self, df: pd.DataFrame, period: int = 14, high_col: str = 'high', low_col: str = 'low', close_col: str = 'close') -> Optional[pd.DataFrame]:
        """
        计算 ATRR (Average True Range Ratio)。
        ATRR = ATR / Close
        """
        required_cols = [high_col, low_col, close_col]
        if df is None or df.empty or not all(c in df.columns for c in required_cols):
            return None
        if len(df) < period:
            return None
        
        try:
            # 1. 计算 ATR
            atr_series = ta.atr(high=df[high_col], low=df[low_col], close=df[close_col], length=period, append=False)
            if atr_series is None or atr_series.empty:
                return None
            
            # 2. 计算 ATRR
            close_prices = df[close_col].replace(0, np.nan) # 避免除以零
            atrr_series = atr_series / close_prices
            
            # 3. 返回DataFrame
            return pd.DataFrame({f'ATRr_{period}': atrr_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 ATRR (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_atrn(self, df: pd.DataFrame, period: int = 14, high_col: str = 'high', low_col: str = 'low', close_col: str = 'close') -> Optional[pd.DataFrame]:
        """
        计算 ATRN (归一化平均真实波幅)。
        此方法先计算ATR，然后手动进行归一化处理 (ATR / Close)。
        """
        # 步骤1: 输入验证。ATRN需要高、低、收三列数据。
        required_cols = [high_col, low_col, close_col]
        if df is None or df.empty or not all(col in df.columns for col in required_cols):
            logger.warning(f"计算 ATRN 失败：DataFrame 中缺少必需的列。需要 {required_cols}。")
            return None
        
        # 步骤2: 数据长度验证。
        if len(df) < period:
            logger.warning(f"计算 ATRN 失败：数据行数 {len(df)} 小于周期 {period}。")
            return None
            
        try:
            # 步骤3: 定义一个内部同步函数来执行计算。
            def _sync_atrn():
                # 核心修改：pandas_ta 没有 atrn，我们先计算 atr
                atr_series = ta.atr(high=df[high_col], low=df[low_col], close=df[close_col], length=period, append=False)
                
                if atr_series is None or atr_series.empty:
                    logger.warning(f"ATRN 计算失败：基础 ATR 计算返回了空结果。")
                    return None

                # 获取收盘价序列，并处理收盘价为0的情况，避免除零错误
                close_prices = df[close_col].replace(0, np.nan)
                
                # 手动进行归一化计算： (ATR / Close) * 100
                atrn_series = (atr_series / close_prices) * 100
                
                # 将计算结果包装成一个DataFrame，并使用标准命名
                return pd.DataFrame({f'ATRN_{period}': atrn_series})
            
            # 步骤4: 异步执行同步函数。
            atrn_df = await asyncio.to_thread(_sync_atrn)
            
            # 步骤5: 检查计算结果是否有效。
            if atrn_df is None or atrn_df.empty:
                logger.warning(f"ATRN 计算返回了空结果。")
                return None
                
            # 步骤6: 返回计算成功的DataFrame。
            return atrn_df
            
        except Exception as e:
            # 步骤7: 捕获并记录异常。
            logger.error(f"计算 ATRN (period={period}) 时出错: {e}", exc_info=True)
            return None

    async def calculate_boll_bands_and_width(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, close_col='close') -> Optional[pd.DataFrame]:
        """
        【V1.1 标准化版】计算布林带 (BBANDS) 及其宽度 (BBW) 和百分比B (%B)
        - 核心修正: 对 pandas-ta 返回的 BBB% (带宽百分比) 列进行标准化，将其除以 100，转换为标准比率。
        """
        if df is None or df.empty or close_col not in df.columns:
            logger.warning(f"计算布林带缺少必要列: {close_col}。")
            return None
        if len(df) < period:
            logger.warning(f"数据行数 ({len(df)}) 不足以计算周期为 {period} 的布林带。")
            return None
        try:
            def _sync_bbands():
                # 使用 ta.bbands() 直接调用，返回一个新的DataFrame
                return ta.bbands(close=df[close_col], length=period, std=std_dev, append=False)

            bbands_df = await asyncio.to_thread(_sync_bbands)
            if bbands_df is None or bbands_df.empty:
                logger.warning(f"布林带 (周期 {period}) 计算结果为空。")
                return None

            # ▼▼▼【核心修正】▼▼▼
            # pandas-ta 返回的 'BBB' 列是百分比形式，我们需要将其转换为标准比率
            # 1. 确定 pandas-ta 输出的原始列名
            bbw_source_col = f'BBB_{period}_{std_dev:.1f}'
            
            # 2. 检查该列是否存在，然后进行标准化
            if bbw_source_col in bbands_df.columns:
                # print(f"  [指标标准化] 检测到 pandas-ta 的 '{bbw_source_col}' 列。")
                # print(f"  [指标标准化] 将其值除以 100.0 以从百分比转换为标准比率。")
                # 核心操作：将百分比转换为比率
                bbands_df[bbw_source_col] = bbands_df[bbw_source_col] / 100.0
            # ▲▲▲【核心修正】▲▲▲

            # 3. 现在可以安全地重命名了，重命名后的 'BBW' 列将包含正确的比率值
            rename_map = {
                f'BBL_{period}_{std_dev:.1f}': f'BBL_{period}_{std_dev:.1f}',
                f'BBM_{period}_{std_dev:.1f}': f'BBM_{period}_{std_dev:.1f}',
                f'BBU_{period}_{std_dev:.1f}': f'BBU_{period}_{std_dev:.1f}',
                bbw_source_col: f'BBW_{period}_{std_dev:.1f}', # 使用源列名进行重命名
                f'BBP_{period}_{std_dev:.1f}': f'BBP_{period}_{std_dev:.1f}'
            }
            
            result_df = bbands_df.rename(columns=rename_map)
            
            # 筛选出我们需要的列
            final_columns = list(rename_map.values())
            result_df = result_df[[col for col in final_columns if col in result_df.columns]]
            
            return result_df if not result_df.empty else None
        except Exception as e:
            logger.error(f"计算布林带及宽度 (周期 {period}, 标准差 {std_dev}) 出错: {e}", exc_info=True)
            return None

    async def calculate_historical_volatility(self, df: pd.DataFrame, period: int = 20, window_type: Optional[str] = None, close_col='close', annual_factor: int = 252) -> Optional[pd.DataFrame]:
        """计算历史波动率 (HV)"""
        if df is None or df.empty or close_col not in df.columns: return None
        if len(df) < period: return None
        try:
            def _sync_hv():
                log_returns = np.log(df[close_col] / df[close_col].shift(1))
                hv_series = log_returns.rolling(window=period, min_periods=max(1, int(period * 0.5))).std() * np.sqrt(annual_factor)
                return pd.DataFrame({f'HV_{period}': hv_series}, index=df.index)
            return await asyncio.to_thread(_sync_hv)
        except Exception as e:
            logger.error(f"计算历史波动率 (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_keltner_channels(self, df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, atr_multiplier: float = 2.0, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算肯特纳通道 (KC)"""
        if df is None or df.empty or not all(col in df.columns for col in [high_col, low_col, close_col]):
            logger.warning(f"计算肯特纳通道缺少必要列。")
            return None
        if len(df) < max(ema_period, atr_period):
            logger.warning(f"数据行数 ({len(df)}) 不足以计算肯特纳通道。")
            return None
        try:
            def _sync_kc():
                # --- 代码修改: 统一使用 ta.kc() 直接调用，其行为更可预测 ---
                return ta.kc(high=df[high_col], low=df[low_col], close=df[close_col], length=ema_period, atr_length=atr_period, scalar=atr_multiplier, mamode="ema", append=False)
            
            kc_df = await asyncio.to_thread(_sync_kc)
            
            if kc_df is None or kc_df.empty:
                logger.warning(f"肯特纳通道 (EMA周期 {ema_period}) 计算结果为空。")
                return None
            # 检查返回的列数是否符合预期
            if kc_df.shape[1] != 3:
                logger.error(f"肯特纳通道计算返回的列数不为3，实际为 {kc_df.shape[1]}。列名: {kc_df.columns.tolist()}")
                return None
            # 构建我们期望的、与原始代码意图完全一致的列名
            target_lower_col = f'KCL_{ema_period}_{atr_period}'
            target_middle_col = f'KCM_{ema_period}_{atr_period}'
            target_upper_col = f'KCU_{ema_period}_{atr_period}'
            # 创建一个新的 DataFrame，使用原始索引和我们期望的列名
            result_df = pd.DataFrame({
                target_lower_col: kc_df.iloc[:, 0], # 第1列是下轨
                target_middle_col: kc_df.iloc[:, 1], # 第2列是中轨
                target_upper_col: kc_df.iloc[:, 2]  # 第3列是上轨
            }, index=df.index)
            return result_df
        except Exception as e:
            logger.error(f"计算肯特纳通道 (EMA周期 {ema_period}, ATR周期 {atr_period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_cci(self, df: pd.DataFrame, period: int = 14, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算 CCI (商品渠道指数)"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]): return None
        if len(df) < period: return None
        try:
            def _sync_cci():
                return ta.cci(high=df[high_col], low=df[low_col], close=df[close_col], length=period, append=False)
            cci_series = await asyncio.to_thread(_sync_cci)
            if cci_series is None or cci_series.empty: return None
            return pd.DataFrame({f'CCI_{period}': cci_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 CCI (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_cmf(self, df: pd.DataFrame, period: int = 20, high_col='high', low_col='low', close_col='close', volume_col='volume') -> Optional[pd.DataFrame]:
        """
        【V2 修正版】计算 CMF (蔡金货币流量)。
        - 修正了列名问题，确保与上层合并逻辑兼容。
        - 简化了异步实现，使其更直接、更健壮。
        """
        required_cols = [high_col, low_col, close_col, volume_col]
        if df is None or df.empty or not all(col in df.columns for col in required_cols):
            logger.warning(f"计算 CMF (周期 {period}) 失败：输入DataFrame为空或缺少必需列 {required_cols}。")
            return None
        
        if len(df) < period:
            logger.warning(f"计算 CMF (周期 {period}) 失败：数据长度 {len(df)} 小于周期 {period}。")
            return None
            
        try:
            # 直接调用 pandas_ta，它会返回一个带有正确列名（如 'CMF_20'）的 Series
            # 我们不需要手动创建 DataFrame 或重命名列
            # print(f"调试信息: [IndicatorService] 正在为周期 {period} 计算 CMF...")
            cmf_series = ta.cmf(
                high=df[high_col], 
                low=df[low_col], 
                close=df[close_col], 
                volume=df[volume_col], 
                length=period, 
                append=False # 确保不修改原始df
            )
            
            if cmf_series is None or cmf_series.empty:
                logger.warning(f"计算 CMF (周期 {period}) 返回了空结果。")
                return None
            
            # print(f"调试信息: [IndicatorService] CMF (周期 {period}) 计算完成，结果类型: {type(cmf_series)}，列名: {cmf_series.name}")
            # 将返回的 Series 转换为 DataFrame，后续的合并逻辑会处理它
            return cmf_series.to_frame()

        except Exception as e:
            logger.error(f"计算 CMF (周期 {period}) 时发生未知异常: {e}", exc_info=True)
            return None

    async def calculate_kdj(self, df: pd.DataFrame, period: int = 9, signal_period: int = 3, smooth_k_period: int = 3, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算KDJ指标"""
        required_cols = [high_col, low_col, close_col]
        if df is None or df.empty or not all(c in df.columns for c in required_cols):
            return None
        if len(df) < period + signal_period:
            return None

        try:
            def _sync_stoch():
                # pandas-ta的stoch函数计算的就是KDJ
                return ta.stoch(high=df[high_col], low=df[low_col], close=df[close_col], k=period, d=signal_period, smooth_k=smooth_k_period, append=False)
            
            stoch_df = await asyncio.to_thread(_sync_stoch)

            # ▼▼▼【修改】增加对 None 返回值的健壮性检查 ▼▼▼
            if stoch_df is None or stoch_df.empty:
                logger.warning(f"KDJ (p={period}, sig={signal_period}, smooth={smooth_k_period}) 计算结果为空，可能数据量不足。")
                return None
            # ▲▲▲ 修改结束 ▲▲▲

            # 重命名列以符合KDJ的习惯
            # STOCHk_9_3_3 -> K_9_3_3, STOCHd_9_3_3 -> D_9_3_3
            stoch_df.rename(columns=lambda x: x.replace('STOCHk', 'K').replace('STOCHd', 'D'), inplace=True)
            
            # 计算J值: J = 3*K - 2*D
            k_col = f'K_{period}_{signal_period}_{smooth_k_period}'
            d_col = f'D_{period}_{signal_period}_{smooth_k_period}'
            j_col = f'J_{period}_{signal_period}_{smooth_k_period}'
            
            if k_col in stoch_df and d_col in stoch_df:
                stoch_df[j_col] = 3 * stoch_df[k_col] - 2 * stoch_df[d_col]
            
            return stoch_df

        except Exception as e:
            logger.error(f"计算 KDJ (p={period}, sig={signal_period}, smooth={smooth_k_period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_ema(self, df: pd.DataFrame, period: int, close_col='close') -> Optional[pd.DataFrame]:
        """计算 EMA (指数移动平均线)"""
        if df is None or df.empty or close_col not in df.columns: return None
        if len(df) < period: return None
        try:
            def _sync_ema():
                return ta.ema(close=df[close_col], length=period, append=False)
            ema_series = await asyncio.to_thread(_sync_ema)
            if ema_series is None or not isinstance(ema_series, pd.Series) or ema_series.empty: return None
            return pd.DataFrame({f'EMA_{period}': ema_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 EMA (周期 {period}) 时发生未知错误: {e}", exc_info=True)
            return None

    async def calculate_dmi(self, df: pd.DataFrame, period: int = 14, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算 DMI (动向指标), 包括 PDI (+DI), NDI (-DI), ADX"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]): return None
        if len(df) < period: return None
        try:
            def _sync_dmi():
                return ta.adx(high=df[high_col], low=df[low_col], close=df[close_col], length=period)
            dmi_df = await asyncio.to_thread(_sync_dmi)
            if dmi_df is None or dmi_df.empty: return None
            rename_map = {
                f'DMP_{period}': f'PDI_{period}',
                f'DMN_{period}': f'NDI_{period}',
                f'ADX_{period}': f'ADX_{period}'
            }
            result_df = dmi_df.rename(columns={k: v for k, v in rename_map.items() if k in dmi_df.columns})
            return result_df
        except Exception as e:
            logger.error(f"计算 DMI (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_ichimoku(self, df: pd.DataFrame, tenkan_period: int = 9, kijun_period: int = 26, senkou_period: int = 52, name_suffix: Optional[str] = None, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """
        计算一目均衡表 (Ichimoku Cloud) 的时间对齐特征。
        Args:
            df (pd.DataFrame): 输入的K线数据。
            tenkan_period (int): 转换线周期。
            kijun_period (int): 基准线周期。
            senkou_period (int): 先行带B周期。
            name_suffix (Optional[str]): 可选的名称后缀，用于附加到所有列名后 (例如 '15', 'D')。
            ...
        Returns:
            Optional[pd.DataFrame]: 包含一目均衡表指标的DataFrame。
        """
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]): return None
        if len(df) < max(tenkan_period, kijun_period, senkou_period): return None
        try:
            def _sync_ichimoku():
                # 使用 ta.ichimoku() 直接调用，获取时间对齐的特征
                # 返回的第一个元素是包含 ITS, IKS, ISA, ISB, ICS 的 DataFrame
                ichimoku_data, _ = ta.ichimoku(high=df[high_col], low=df[low_col], close=df[close_col],
                                           tenkan=tenkan_period, kijun=kijun_period, senkou=senkou_period,
                                           append=False)
                return ichimoku_data
            ichi_df = await asyncio.to_thread(_sync_ichimoku)
            if ichi_df is None or ichi_df.empty: return None
            # 源列名 (由 pandas-ta 生成)
            source_tenkan = f'ITS_{tenkan_period}'
            source_kijun = f'IKS_{kijun_period}'
            source_senkou_a = f'ISA_{tenkan_period}'
            source_senkou_b = f'ISB_{kijun_period}' # 注意: pandas-ta 使用 kijun 周期命名
            source_chikou = f'ICS_{kijun_period}'   # 注意: pandas-ta 使用 kijun 周期命名
            # 目标列名 (基础部分，与原始代码意图一致)
            target_tenkan = f'TENKAN_{tenkan_period}'
            target_kijun = f'KIJUN_{kijun_period}'
            target_senkou_a = f'SENKOU_A_{tenkan_period}'
            target_senkou_b = f'SENKOU_B_{senkou_period}' # 目标名使用 senkou 周期，更符合逻辑
            target_chikou = f'CHIKOU_{kijun_period}'
            rename_map = {
                source_tenkan: target_tenkan,
                source_kijun: target_kijun,
                source_senkou_a: target_senkou_a,
                source_senkou_b: target_senkou_b,
                source_chikou: target_chikou,
            }
            
            result_df = ichi_df.rename(columns=rename_map)
            # 筛选出我们成功重命名的列，避免携带非预期的列
            final_columns = list(rename_map.values())
            result_df = result_df[[col for col in final_columns if col in result_df.columns]]
            # 如果提供了后缀，则附加到所有列名上
            if name_suffix:
                result_df.columns = [f'{col}_{name_suffix}' for col in result_df.columns]
            return result_df if not result_df.empty else None
        except Exception as e:
            logger.error(f"计算 Ichimoku (t={tenkan_period}, k={kijun_period}, s={senkou_period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_sma(self, df: pd.DataFrame, period: int, close_col='close') -> Optional[pd.DataFrame]:
        """计算 SMA (简单移动平均线)"""
        if df is None or df.empty or close_col not in df.columns: return None
        if len(df) < period: return None
        try:
            def _sync_sma():
                return ta.sma(close=df[close_col], length=period, append=False)
            sma_series = await asyncio.to_thread(_sync_sma)
            if sma_series is None or not isinstance(sma_series, pd.Series) or sma_series.empty: return None
            return pd.DataFrame({f'SMA_{period}': sma_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 SMA (周期 {period}) 时发生未知错误: {e}", exc_info=True)
            return None

    async def calculate_amount_ma(self, df: pd.DataFrame, period: int = 20, amount_col='amount') -> Optional[pd.DataFrame]:
        """计算成交额的移动平均线 (AMT_MA)"""
        if df is None or df.empty or amount_col not in df.columns: return None
        if len(df) < period: return None
        try:
            def _sync_amount_ma():
                return df[amount_col].rolling(window=period, min_periods=max(1, int(period*0.5))).mean()
            amt_ma_series = await asyncio.to_thread(_sync_amount_ma)
            return pd.DataFrame({f'AMT_MA_{period}': amt_ma_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 AMT_MA (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_macd(self, df: pd.DataFrame, period_fast: int = 12, period_slow: int = 26, signal_period: int = 9, close_col='close') -> Optional[pd.DataFrame]:
        """计算移动平均收敛散度 (MACD)"""
        if df is None or df.empty or close_col not in df.columns:
            return None
        if len(df) < period_slow + signal_period:
            return None

        try:
            def _sync_macd():
                return ta.macd(close=df[close_col], fast=period_fast, slow=period_slow, signal=signal_period, append=False)
            
            macd_df = await asyncio.to_thread(_sync_macd)

            # ▼▼▼【修改】增加对 None 返回值的健壮性检查 ▼▼▼
            if macd_df is None or macd_df.empty:
                logger.warning(f"MACD (f={period_fast},s={period_slow},sig={signal_period}) 计算结果为空，可能数据量不足。")
                return None
            # ▲▲▲ 修改结束 ▲▲▲
            
            return macd_df

        except Exception as e:
            logger.error(f"计算 MACD (f={period_fast},s={period_slow},sig={signal_period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_mfi(self, df: pd.DataFrame, period: int = 14, high_col='high', low_col='low', close_col='close', volume_col='volume') -> Optional[pd.DataFrame]:
        """计算 MFI (资金流量指标)"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col, volume_col]): return None
        if len(df) < period: return None
        try:
            def _sync_mfi():
                return ta.mfi(high=df[high_col], low=df[low_col], close=df[close_col], volume=df[volume_col], length=period, append=False)
            mfi_series = await asyncio.to_thread(_sync_mfi)
            if mfi_series is None or mfi_series.empty: return None
            return pd.DataFrame({f'MFI_{period}': mfi_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 MFI (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_mom(self, df: pd.DataFrame, period: int, close_col='close') -> Optional[pd.DataFrame]:
        """计算 MOM (动量指标)"""
        if df is None or df.empty or close_col not in df.columns: return None
        if len(df) < period: return None
        try:
            def _sync_mom():
                return ta.mom(close=df[close_col], length=period)
            mom_series = await asyncio.to_thread(_sync_mom)
            if mom_series is None or mom_series.empty: return None
            return pd.DataFrame({f'MOM_{period}': mom_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 MOM (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_obv(self, df: pd.DataFrame, close_col='close', volume_col='volume') -> Optional[pd.DataFrame]:
        """计算 OBV (能量潮指标)"""
        if df is None or df.empty or not all(c in df.columns for c in [close_col, volume_col]): return None
        try:
            def _sync_obv():
                return ta.obv(close=df[close_col], volume=df[volume_col], append=False)
            obv_series = await asyncio.to_thread(_sync_obv)
            if obv_series is None or obv_series.empty: return None
            return pd.DataFrame({'OBV': obv_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 OBV 出错: {e}", exc_info=True)
            return None

    async def calculate_roc(self, df: pd.DataFrame, period: int = 12, close_col='close') -> Optional[pd.DataFrame]:
        """计算 ROC (价格变化率)"""
        if df is None or df.empty or close_col not in df.columns: return None
        if len(df) <= period: return None
        try:
            def _sync_roc():
                return ta.roc(close=df[close_col], length=period, append=False)
            roc_series = await asyncio.to_thread(_sync_roc)
            if roc_series is None or roc_series.empty: return None
            return pd.DataFrame({f'ROC_{period}': roc_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 ROC (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_amount_roc(self, df: pd.DataFrame, period: int, amount_col='amount') -> Optional[pd.DataFrame]:
        """计算成交额的 ROC (AROC)"""
        if df is None or df.empty or amount_col not in df.columns:
            logger.warning(f"输入 DataFrame 为空或缺少 '{amount_col}' 列，无法计算 AROC。")
            return None
        # --- 调试信息：打印输入DataFrame的形状和列名 ---
        # print(f"调试信息: [AROC_{period}] 输入 df 的形状: {df.shape}, 列: {df.columns.tolist()}")
        if len(df) <= period:
            logger.warning(f"数据行数 ({len(df)}) 不足以计算周期为 {period} 的 AROC。")
            return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_aroc():
                target_series = df[amount_col]
                # 直接调用 ta.roc 函数，传入 Series
                return ta.roc(close=target_series, length=period, append=False)
                # --- 代码修改结束 ---
            aroc_series = await asyncio.to_thread(_sync_aroc)
            if aroc_series is None or aroc_series.empty:
                logger.warning(f"AROC_{period} 计算结果为空。")
                return None
            # 将结果构建为 DataFrame，列名格式化
            df_results = pd.DataFrame({f'AROC_{period}': aroc_series})
            # 将无穷大值替换为NaN，便于后续处理
            df_results.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_results
        except Exception as e:
            # 记录详细的错误信息，包括堆栈跟踪
            logger.error(f"计算 Amount ROC (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_volume_roc(self, df: pd.DataFrame, period: int, volume_col='volume') -> Optional[pd.DataFrame]:
        """计算成交量的 ROC (VROC)"""
        if df is None or df.empty or volume_col not in df.columns:
            logger.warning(f"输入 DataFrame 为空或缺少 '{volume_col}' 列，无法计算 VROC。")
            return None
        if len(df) <= period:
            logger.warning(f"数据行数 ({len(df)}) 不足以计算周期为 {period} 的 VROC。")
            return None
            
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_vroc():
                target_series = df[volume_col]
                # 直接调用 ta.roc 函数，传入 Series。注意：pandas_ta 的 roc 函数使用 'close' 作为通用输入参数名。
                return ta.roc(close=target_series, length=period)
            vroc_series = await asyncio.to_thread(_sync_vroc)
            if vroc_series is None or vroc_series.empty:
                logger.warning(f"VROC_{period} 计算结果为空。")
                return None
            df_results = pd.DataFrame({f'VROC_{period}': vroc_series}, index=df.index)
            # 将无穷大值替换为NaN，便于后续处理
            df_results.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_results
        except Exception as e:
            logger.error(f"计算 Volume ROC (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_rsi(self, df: pd.DataFrame, period: int, close_col='close') -> Optional[pd.DataFrame]:
        """计算相对强弱指数 (RSI)"""
        if df is None or df.empty or close_col not in df.columns:
            return None
        if len(df) < period:
            return None
        
        try:
            # 异步执行 pandas_ta 计算
            def _sync_rsi():
                return ta.rsi(close=df[close_col], length=period, append=False)
            rsi_series = await asyncio.to_thread(_sync_rsi)

            # ▼▼▼【修改】增加对 None 返回值的健壮性检查 ▼▼▼
            if rsi_series is None or not isinstance(rsi_series, pd.Series) or rsi_series.empty:
                logger.warning(f"RSI (周期 {period}) 计算结果为空或无效，可能数据量不足。")
                return None
            
            # 【修改】在创建DataFrame时显式传入索引，更加安全
            return pd.DataFrame({f'RSI_{period}': rsi_series}, index=df.index)
            # ▲▲▲ 修改结束 ▲▲▲

        except Exception as e:
            logger.error(f"计算 RSI (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_sar(self, df: pd.DataFrame, af_step: float = 0.02, max_af: float = 0.2, high_col='high', low_col='low') -> Optional[pd.DataFrame]:
        """计算 SAR (抛物线转向指标)"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col]): return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_psar():
                return df.ta.psar(high=df[high_col], low=df[low_col], af0=af_step, af=af_step, max_af=max_af, append=False)
            psar_df = await asyncio.to_thread(_sync_psar)
            if psar_df is None or psar_df.empty: return None
            long_sar_col = next((col for col in psar_df.columns if col.startswith('PSARl')), None)
            short_sar_col = next((col for col in psar_df.columns if col.startswith('PSARs')), None)
            if long_sar_col and short_sar_col:
                sar_values = psar_df[long_sar_col].fillna(psar_df[short_sar_col])
                return pd.DataFrame({f'SAR_{af_step:.2f}_{max_af:.2f}': sar_values})
            elif long_sar_col:
                return pd.DataFrame({f'SAR_{af_step:.2f}_{max_af:.2f}': psar_df[long_sar_col]})
            elif short_sar_col:
                return pd.DataFrame({f'SAR_{af_step:.2f}_{max_af:.2f}': psar_df[short_sar_col]})
            else:
                logger.warning(f"计算 SAR 未找到 PSARl 或 PSARs 列。返回列: {psar_df.columns.tolist()}")
                return None
        except Exception as e:
            logger.error(f"计算 SAR (af={af_step:.2f}, max_af={max_af:.2f}) 出错: {e}", exc_info=True)
            return None

    async def calculate_stoch(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3, smooth_k_period: int = 3, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算 STOCH (随机指标)"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]): return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_stoch():
                return df.ta.stoch(high=df[high_col], low=df[low_col], close=df[close_col], k=k_period, d=d_period, smooth_k=smooth_k_period, append=False)
            stoch_df = await asyncio.to_thread(_sync_stoch)
            if stoch_df is None or stoch_df.empty: return None
            return stoch_df
        except Exception as e:
            logger.error(f"计算 STOCH (k={k_period},d={d_period},s={smooth_k_period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_adl(self, df: pd.DataFrame, high_col='high', low_col='low', close_col='close', volume_col='volume') -> Optional[pd.DataFrame]:
        """计算 ADL (累积/派发线)"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col, volume_col]):
            return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_adl():
                return df.ta.ad(high=df[high_col], low=df[low_col], close=df[close_col], volume=df[volume_col], append=False)
            adl_series = await asyncio.to_thread(_sync_adl)
            if adl_series is None or adl_series.empty: return None
            return pd.DataFrame({'ADL': adl_series})
        except Exception as e:
            logger.error(f"计算 ADL 出错: {e}", exc_info=True)
            return None

    async def calculate_pivot_points(self, df: pd.DataFrame, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算经典枢轴点和斐波那契枢轴点 (基于前一周期数据)"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]):
            return None
        try:
            # --- 将同步的计算逻辑移至线程中执行 ---
            def _sync_pivot():
                results = pd.DataFrame(index=df.index)
                prev_high = df[high_col].shift(1)
                prev_low = df[low_col].shift(1)
                prev_close = df[close_col].shift(1)
                PP = (prev_high + prev_low + prev_close) / 3
                results['PP'] = PP
                results['S1'] = (2 * PP) - prev_high
                results['S2'] = PP - (prev_high - prev_low)
                results['S3'] = results['S1'] - (prev_high - prev_low)
                results['S4'] = results['S2'] - (prev_high - prev_low)
                results['R1'] = (2 * PP) - prev_low
                results['R2'] = PP + (prev_high - prev_low)
                results['R3'] = results['R1'] + (prev_high - prev_low)
                results['R4'] = results['R2'] + (prev_high - prev_low)
                diff = prev_high - prev_low
                results['F_R1'] = PP + 0.382 * diff
                results['F_R2'] = PP + 0.618 * diff
                results['F_R3'] = PP + 1.000 * diff
                results['F_S1'] = PP - 0.382 * diff
                results['F_S2'] = PP - 0.618 * diff
                results['F_S3'] = PP - 1.000 * diff
                return results.iloc[1:]
            return await asyncio.to_thread(_sync_pivot)
        except Exception as e:
            logger.error(f"计算 Pivot Points 出错: {e}", exc_info=True)
            return None

    async def calculate_vol_ma(self, df: pd.DataFrame, period: int = 20, volume_col='volume') -> Optional[pd.DataFrame]:
        """计算成交量的移动平均线 (VOL_MA)"""
        if df is None or df.empty or volume_col not in df.columns: return None
        try:
            # --- 将同步的 rolling 计算移至线程中执行 ---
            def _sync_vol_ma():
                return df[volume_col].rolling(window=period, min_periods=max(1, int(period*0.5))).mean()
            vol_ma_series = await asyncio.to_thread(_sync_vol_ma)
            return pd.DataFrame({f'VOL_MA_{period}': vol_ma_series})
        except Exception as e:
            logger.error(f"计算 VOL_MA (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_vwap(self, df: pd.DataFrame, anchor: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        【V1.1 锚点修正版】计算 VWAP (成交量加权平均价)。
        - 修正了对分钟级别锚点（如 '30', '60'）的处理，将其转换为 pandas 可识别的频率字符串（如 '30T'）。
        """
        # pandas-ta 需要标准的列名
        high_col, low_col, close_col, volume_col = 'high', 'low', 'close', 'volume'
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col, volume_col]):
            logger.warning(f"计算 VWAP (anchor={anchor}) 时缺少必要的列。")
            return None
        
        # ▼▼▼ 转换分钟级别锚点为pandas兼容格式 ▼▼▼
        # 解释: pandas-ta的vwap函数要求锚点(anchor)是pandas的频率字符串。
        # 对于分钟级别，'30' 是无效的，必须是 '30T' 或 '30min'。
        # 此处对纯数字的锚点进行转换，而 'D', 'W' 等则保持不变。
        processed_anchor = anchor
        if anchor and str(anchor).isdigit():
            processed_anchor = f"{anchor}T"
            # print(f"  [VWAP 调试] 将数字锚点 '{anchor}' 转换为 pandas 频率 '{processed_anchor}'")
        

        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_vwap():
                # ▼▼▼ 使用处理后的锚点 ▼▼▼
                # 调用 pandas_ta 的 vwap 方法
                return df.ta.vwap(high=df[high_col], low=df[low_col], close=df[close_col], volume=df[volume_col], anchor=processed_anchor, append=False)
                
            
            vwap_series = await asyncio.to_thread(_sync_vwap)
            
            if vwap_series is None or vwap_series.empty:
                return None
            
            # pandas_ta 会根据 anchor 自动生成列名，例如 VWAP_D, VWAP_W, VWAP_30T
            # 我们直接使用它返回的 Series，其 name 就是列名
            return pd.DataFrame(vwap_series)
        except Exception as e:
            # ▼▼▼ 在日志中也使用原始锚点，方便追溯 ▼▼▼
            logger.error(f"计算 VWAP (anchor={anchor}) 出错: {e}", exc_info=True)
            
            return None

    async def calculate_willr(self, df: pd.DataFrame, period: int = 14, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算威廉姆斯 %R (WILLR)"""
        required_cols = [high_col, low_col, close_col]
        if df is None or df.empty or not all(c in df.columns for c in required_cols):
            logger.warning(f"输入 DataFrame 为空或缺少必要的列 {required_cols}，无法计算 WILLR。")
            return None
        if len(df) < period:
            logger.warning(f"数据行数 ({len(df)}) 不足以计算周期为 {period} 的 WILLR。")
            return None
            
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_willr():
                return ta.willr(high=df[high_col], low=df[low_col], close=df[close_col], length=period)
            willr_series = await asyncio.to_thread(_sync_willr)
            if willr_series is None or willr_series.empty:
                logger.warning(f"WILLR_{period} 计算结果为空。")
                return None
            df_results = pd.DataFrame({f'WILLR_{period}': willr_series})            
            return df_results
        except Exception as e:
            logger.error(f"计算 WILLR (周期 {period}) 出错: {e}", exc_info=True)
            return None

    def calculate_relative_strength(self, df: pd.DataFrame, stock_close_col: str, benchmark_codes: List[str], periods: List[int], time_level: str) -> pd.DataFrame:
        """
        计算股票相对于基准指数/板块的相对强度/超额收益。
        使用对数收益率的累积差值进行计算。此为同步函数。
        Args:
            df (pd.DataFrame): 包含股票和基准数据的 DataFrame。
            stock_close_col (str): 股票收盘价列名 (已带时间级别后缀)。
            benchmark_codes (List[str]): 基准指数/板块代码列表。
            periods (List[int]): 计算相对强度的周期列表。
            time_level (str): 当前计算的时间级别。
        Returns:
            pd.DataFrame: 补充了相对强度特征的 DataFrame。
        """
        if df is None or df.empty or stock_close_col not in df.columns:
            logger.warning(f"计算相对强度失败，输入 DataFrame 无效或缺少股票收盘价列 {stock_close_col}。")
            print(f"计算相对强度失败，输入 DataFrame 无效或缺少股票收盘价列 {stock_close_col}。")
            return df
        df_processed = df.copy()
        stock_close_shifted = df_processed[stock_close_col].shift(1)
        stock_returns = np.log(df_processed[stock_close_col] / stock_close_shifted)
        stock_returns = stock_returns.replace([np.inf, -np.inf], np.nan)
        for benchmark_code in benchmark_codes:
            if '.' in benchmark_code:
                benchmark_col_prefix = f'index_{benchmark_code.replace(".", "_").lower()}_'
            else:
                benchmark_col_prefix = f'ths_{benchmark_code.replace(".", "_").lower()}_'
            benchmark_close_col = f'{benchmark_col_prefix}close'
            if benchmark_close_col in df_processed.columns:
                benchmark_close_shifted = df_processed[benchmark_close_col].shift(1)
                benchmark_returns = np.log(df_processed[benchmark_close_col] / benchmark_close_shifted)
                benchmark_returns = benchmark_returns.replace([np.inf, -np.inf], np.nan)
                for period in periods:
                    cumulative_stock_log_return = stock_returns.rolling(window=period).sum()
                    cumulative_benchmark_log_return = benchmark_returns.rolling(window=period).sum()
                    excess_log_return = cumulative_stock_log_return - cumulative_benchmark_log_return
                    rs_col_name = f'RS_{benchmark_code.replace(".", "_").lower()}_{period}_{time_level}'
                    df_processed[rs_col_name] = excess_log_return
        return df_processed

    async def calculate_trix(self, df: pd.DataFrame, period: int = 14, signal_period: int = 9, close_col='close') -> Optional[pd.DataFrame]:
        """
        计算 TRIX (三重指数平滑移动平均线) 及其信号线。
        Args:
            df (pd.DataFrame): 输入数据。
            period (int): TRIX 计算周期。
            signal_period (int): 信号线计算周期。
            close_col (str): 收盘价列名。
        Returns:
            Optional[pd.DataFrame]: 包含 TRIX 和 TRIX_signal 列的 DataFrame。
        """
        if df is None or df.empty or close_col not in df.columns:
            return None
        if len(df) < period * 3: # TRIX 需要更长的启动数据
            logger.warning(f"数据行数 ({len(df)}) 不足以计算周期为 {period} 的 TRIX。")
            return None
        try:
            def _sync_trix():
                # 使用 pandas-ta 直接计算 TRIX 和其信号线
                return ta.trix(close=df[close_col], length=period, signal=signal_period, append=False)
            trix_df = await asyncio.to_thread(_sync_trix)
            if trix_df is None or trix_df.empty:
                return None
            # pandas-ta 默认返回的列名是 'TRIX_14_9' 和 'TRIXs_14_9'，这已经很清晰，直接返回即可
            return trix_df
        except Exception as e:
            logger.error(f"计算 TRIX (period={period}, signal={signal_period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_coppock(self, df: pd.DataFrame, long_roc_period: int = 26, short_roc_period: int = 13, wma_period: int = 10) -> Optional[pd.DataFrame]:
        """
        【V1.3 健壮性修复版】计算 Coppock Curve (COPP) 指标。
        - 核心修复: 解决了 'Series' object has no attribute 'columns' 的崩溃问题。
                    通过检查返回值类型，并在其为 Series 时使用 .to_frame() 将其统一转换为
                    DataFrame，使后续的重命名逻辑对两种返回类型都能兼容。
        """
        if df is None or df.empty or 'close' not in df.columns:
            return None
        try:
            copp_df = df.ta.coppock(
                close=df['close'],
                length=wma_period,
                fast=short_roc_period,
                slow=long_roc_period,
                append=False
            )
            if copp_df is not None and not copp_df.empty:
                # ▼▼▼【代码修改】: 核心修复，兼容Series和DataFrame返回值 ▼▼▼
                # 检查返回的是否是Series，如果是，则转换为DataFrame，以统一处理
                if isinstance(copp_df, pd.Series):
                    copp_df = copp_df.to_frame()

                # 现在 copp_df 肯定是DataFrame，可以安全地访问 .columns
                if not copp_df.columns[0].startswith('COPP'):
                    expected_name = f"COPP_{long_roc_period}_{short_roc_period}_{wma_period}"
                    actual_name = copp_df.columns[0]
                    copp_df.rename(columns={actual_name: expected_name}, inplace=True)
                    # print(f"    - [指标重命名] 已将列 '{actual_name}' 重命名为 '{expected_name}'")
                # ▲▲▲【代码修改】: 修改结束 ▲▲▲
                return copp_df
        except Exception as e:
            # 增加数据量不足的特定警告
            if "data length" in str(e).lower() or "inputs are all nan" in str(e).lower():
                logger.warning(f"数据行数 ({len(df)}) 不足以计算 Coppock Curve(long={long_roc_period}, short={short_roc_period}, wma={wma_period})。")
            else:
                # 在日志中包含异常类型，方便调试
                logger.error(f"计算 Coppock Curve 时发生未知错误: {type(e).__name__}: {e}", exc_info=False) # exc_info=False 避免打印完整堆栈
        
        return None

    async def calculate_uo(self, df: pd.DataFrame, short_period: int = 7, medium_period: int = 14, long_period: int = 28, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """
        计算 Ultimate Oscillator (终极波动指标)。
        Args:
            df (pd.DataFrame): 输入数据。
            short_period (int): 短周期。
            medium_period (int): 中周期。
            long_period (int): 长周期。
            ...
        Returns:
            Optional[pd.DataFrame]: 包含 UO 指标的 DataFrame。
        """
        required_cols = [high_col, low_col, close_col]
        if df is None or df.empty or not all(c in df.columns for c in required_cols):
            return None
        if len(df) < long_period:
            logger.warning(f"数据行数 ({len(df)}) 不足以计算周期为 {long_period} 的 UO。")
            return None
        try:
            def _sync_uo():
                # 使用 pandas-ta 直接计算
                return ta.uo(high=df[high_col], low=df[low_col], close=df[close_col], fast=short_period, medium=medium_period, slow=long_period, append=False)
            uo_series = await asyncio.to_thread(_sync_uo)
            if uo_series is None or uo_series.empty:
                return None
            # 返回一个标准的 DataFrame
            col_name = f'UO_{short_period}_{medium_period}_{long_period}'
            return pd.DataFrame({col_name: uo_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 Ultimate Oscillator 出错: {e}", exc_info=True)
            return None

    async def calculate_bias(self, df: pd.DataFrame, period: int = 20, close_col='close') -> Optional[pd.DataFrame]:
        """
        【V1.3 最终修正版】计算 BIAS，并强制重命名列以符合系统标准。
        """
        # (此函数前面的调试代码和检查代码保持不变)
        if df is None or df.empty or close_col not in df.columns:
            logger.warning(f"BIAS计算失败：输入的DataFrame为空或缺少'{close_col}'列。")
            return None
        
        if len(df) < period:
            logger.warning(f"BIAS计算失败：数据长度 {len(df)} 小于所需周期 {period}。")
            return None

        try:
            def _sync_bias() -> Optional[pd.Series]:
                # pandas_ta.bias 会生成一个名为 'BIAS_SMA_{period}' 的列
                return df.ta.bias(close=df[close_col], length=period, append=False)

            bias_series = await asyncio.to_thread(_sync_bias)

            if bias_series is None or bias_series.empty:
                logger.warning(f"pandas_ta.bias 未能为周期 {period} 生成有效结果。")
                return None

            # 这是解决问题的核心：将 pandas_ta 生成的列名 'BIAS_SMA_20' 重命名为我们需要的标准格式 'BIAS_20'
            target_col_name = f"BIAS_{period}"
            bias_series.name = target_col_name

            # 将重命名后的 Series 转换为 DataFrame
            result_df = pd.DataFrame(bias_series)
            return result_df

        except Exception as e:
            logger.error(f"计算 BIAS (period={period}) 时发生未知错误: {e}", exc_info=True)
            return None
    
    async def calculate_consolidation_period(self, df: pd.DataFrame, params: Dict, suffix: str) -> Optional[pd.DataFrame]:
        """
        【V2.2 NaN根除版】根据多因子共振识别盘整期。
        - 核心修复1: 保持对 dynamic_bbw_threshold 的 bfill()，解决动态阈值NaN问题。
        - 核心修复2: 在所有计算和ffill之后，对箱体高低点列中残余的NaN（通常在数据开头）
                    用当期的high/low进行填充，从根源上消除NaN，确保下游策略总能获得有效数值。
        """
        # 1. 参数获取 (无变化)
        boll_period = params.get('boll_period', 20)
        boll_std = params.get('boll_std', 2.0)
        bbw_quantile = params.get('bbw_quantile', 0.25)
        roc_period = params.get('roc_period', 12)
        roc_threshold = params.get('roc_threshold', 5.0)
        vol_ma_period = params.get('vol_ma_period', 55)
        min_expanding_periods = boll_period * 2

        # 2. 依赖列名 (无变化)
        bbw_col = f"BBW_{boll_period}_{float(boll_std)}"
        roc_col = f"ROC_{roc_period}"
        vol_ma_col = f"VOL_MA_{vol_ma_period}"
        
        # 3. 依赖检查 (无变化)
        required_cols = [bbw_col, roc_col, vol_ma_col, 'high', 'low', 'volume']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"    - [依赖错误] V2.2箱体计算跳过，依赖的列 '{', '.join(missing)}{suffix}' 不存在。")
            return None

        # 4. 初始化结果DataFrame (无变化)
        result_df = pd.DataFrame(index=df.index)
        output_cols = [
            'is_consolidating', 'dynamic_bbw_threshold', 'dynamic_consolidation_high', 
            'dynamic_consolidation_low', 'dynamic_consolidation_avg_vol', 'dynamic_consolidation_duration'
        ]
        for col in output_cols:
            result_df[col] = np.nan if col not in ['is_consolidating'] else False

        # 5. 核心逻辑 (无变化)
        dynamic_bbw_threshold = df[bbw_col].expanding(min_periods=min_expanding_periods).quantile(bbw_quantile)
        dynamic_bbw_threshold.bfill(inplace=True)
        result_df['dynamic_bbw_threshold'] = dynamic_bbw_threshold
        cond_volatility = df[bbw_col] < result_df['dynamic_bbw_threshold']
        cond_trend = df[roc_col].abs() < roc_threshold
        cond_volume = df['volume'] < df[vol_ma_col]
        is_consolidating = cond_volatility & cond_trend & cond_volume
        result_df['is_consolidating'] = is_consolidating

        if is_consolidating.any():
            # 6. 计算箱体指标 (无变化)
            consolidation_blocks = (is_consolidating != is_consolidating.shift()).cumsum()
            consolidating_df = df[is_consolidating].copy()
            grouped = consolidating_df.groupby(consolidation_blocks[is_consolidating])
            consolidation_high = grouped['high'].transform('max')
            consolidation_low = grouped['low'].transform('min')
            consolidation_avg_vol = grouped['volume'].transform('mean')
            consolidation_duration = grouped['high'].transform('size')

            # 7. 填充结果 (无变化)
            result_df['dynamic_consolidation_high'].update(consolidation_high)
            result_df['dynamic_consolidation_low'].update(consolidation_low)
            result_df['dynamic_consolidation_avg_vol'].update(consolidation_avg_vol)
            result_df['dynamic_consolidation_duration'].update(consolidation_duration)

            fill_cols = [
                'dynamic_consolidation_high', 'dynamic_consolidation_low', 
                'dynamic_consolidation_avg_vol', 'dynamic_consolidation_duration'
            ]
            result_df[fill_cols] = result_df[fill_cols].ffill()

        # 解释: 对于序列开头从未形成过箱体的部分，其 high/low 值为 NaN。
        # 我们用当期自己的 high/low 来填充，确保下游策略总能获得有效的数值进行比较。
        result_df['dynamic_consolidation_high'].fillna(df['high'], inplace=True)
        result_df['dynamic_consolidation_low'].fillna(df['low'], inplace=True)
        # 对于成交量和持续时间，用0填充是合理的默认值
        result_df['dynamic_consolidation_avg_vol'].fillna(0, inplace=True)
        result_df['dynamic_consolidation_duration'].fillna(0, inplace=True)
        
        return result_df

    async def calculate_advanced_fund_features(self, df: pd.DataFrame, params: dict, suffix: str) -> Optional[pd.DataFrame]:
        """
        【V1.0】计算基于资金流和筹码的衍生特征。
        - 核心逻辑: 对原始资金流和筹码数据进行二次加工，生成趋势和偏离度等特征。
        - 依赖项: 
            - 资金流数据 (e.g., 'buy_lg_amount')
            - 筹码数据 (e.g., 'weight_avg')
            - 基础行情数据 ('close')
        - 输出: 
            - fund_buy_lg_amount_ma5: 大单净买入5日移动平均
            - fund_buy_lg_amount_ma10: 大单净买入10日移动平均
            - chip_cost_deviation: 收盘价相对筹码平均成本的偏离度
        """
        # 1. 检查依赖列是否存在于传入的DataFrame中
        #    这是关键一步，确保数据合并已在上游完成
        required_cols = ['buy_lg_amount', 'weight_avg', 'close']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"    - [依赖错误] 资金流衍生特征计算跳过，因缺失必要列: {missing}。请确保上游数据已正确合并。")
            return None

        try:
            # 2. 初始化一个空的DataFrame用于存放结果
            derived_features = pd.DataFrame(index=df.index)

            # 3. 从params或默认值获取参数
            ma_periods = params.get('ma_periods', [5, 10])

            # 4. 计算资金流特征：大单净买入的移动平均
            #    使用循环，更具扩展性
            for period in ma_periods:
                # 注意：输出列名不带后缀，由调用者统一添加
                derived_features[f'fund_buy_lg_amount_ma{period}'] = df['buy_lg_amount'].rolling(window=period).mean()

            # 5. 计算筹码特征：收盘价与平均成本的偏离度
            #    (收盘价 / 平均成本) - 1。 >0 代表股价在成本之上，<0 代表股价在成本之下。
            #    使用 .replace(0, np.nan) 避免除以零的错误
            cost_basis = df['weight_avg'].replace(0, np.nan)
            derived_features['chip_cost_deviation'] = df['close'] / cost_basis - 1
            
            print("    - [信息] 已成功计算资金流和筹码衍生特征。")
            return derived_features

        except Exception as e:
            # 使用Django的logger或您自己的日志系统
            print(f"    - [严重错误] 计算资金流和筹码衍生特征时发生意外: {e}")
            return None

    async def calculate_fibonacci_levels(self, df: pd.DataFrame, params: dict, **kwargs) -> Optional[pd.DataFrame]:
        """
        【新增】计算斐波那契回撤和扩展水平。
        通过识别重要的波段高点和低点，动态计算出潜在的支撑和阻力位。
        """
        fib_params = params.get('fibonacci_analysis_params', {})
        if not fib_params.get('enabled', False):
            return None

        print("    - [斐波那契分析] 开始计算斐波那契水平...")
        
        # 使用scipy.signal.find_peaks识别波段高低点
        distance = fib_params.get('peak_distance', 13)
        prominence = fib_params.get('peak_prominence', 0.05)
        
        # 为了能传入 prominence 序列，我们需要在线程中执行
        def _find_peaks_sync(data, prominence_series):
            # 找出所有候选波峰/波谷
            candidate_indices, _ = find_peaks(data, distance=distance)
            if len(candidate_indices) == 0:
                return []
            # 计算实际突起并与动态阈值比较
            actual_prominences, _, _ = peak_prominences(data, candidate_indices)
            custom_thresholds = prominence_series.iloc[candidate_indices]
            valid_mask = actual_prominences >= custom_thresholds.values
            return candidate_indices[valid_mask]

        # 准备动态prominence阈值
        peak_prominence_series = df['close'] * prominence
        trough_prominence_series = df['close'] * prominence

        # 在线程中异步执行
        peak_indices = await asyncio.to_thread(_find_peaks_sync, df['close'], peak_prominence_series)
        trough_indices = await asyncio.to_thread(_find_peaks_sync, -df['close'], trough_prominence_series)

        # 创建记录最新高低点的列
        df['swing_high_price'] = np.nan
        df.iloc[peak_indices, df.columns.get_loc('swing_high_price')] = df['close'].iloc[peak_indices]
        df['swing_high_price'].ffill(inplace=True)

        df['swing_low_price'] = np.nan
        df.iloc[trough_indices, df.columns.get_loc('swing_low_price')] = df['close'].iloc[trough_indices]
        df['swing_low_price'].ffill(inplace=True)
        
        # 确定当前波段是上升还是下降
        df['swing_high_date'] = pd.NaT
        df.iloc[peak_indices, df.columns.get_loc('swing_high_date')] = df.index[peak_indices]
        df['swing_high_date'].ffill(inplace=True)
        
        df['swing_low_date'] = pd.NaT
        df.iloc[trough_indices, df.columns.get_loc('swing_low_date')] = df.index[trough_indices]
        df['swing_low_date'].ffill(inplace=True)

        # 如果高点比低点新，则当前处于上升波段后的回撤阶段
        is_uptrend_pullback = df['swing_high_date'] > df['swing_low_date']
        
        # 计算波段范围
        swing_range = abs(df['swing_high_price'] - df['swing_low_price'])

        result_df = pd.DataFrame(index=df.index)
        
        # 计算回撤位
        retr_levels = fib_params.get('retracement_levels', [])
        for level in retr_levels:
            col_name = f'fib_retr_{str(level).replace(".", "")}'
            # 上升波段的回撤位 = 高点 - 范围 * 百分比
            retr_price = df['swing_high_price'] - swing_range * level
            result_df[col_name] = np.where(is_uptrend_pullback, retr_price, np.nan)

        # 计算扩展位
        ext_levels = fib_params.get('extension_levels', [])
        for level in ext_levels:
            col_name = f'fib_ext_{str(level).replace(".", "")}'
            # 上升波段的扩展位 = 高点 + 范围 * (扩展百分比 - 1)
            ext_price = df['swing_high_price'] + swing_range * (level - 1.0)
            result_df[col_name] = np.where(is_uptrend_pullback, ext_price, np.nan)
            
        print("    - [斐波那契分析] 计算完成。")
        # 清理临时列
        df.drop(columns=['swing_high_price', 'swing_low_price', 'swing_high_date', 'swing_low_date'], inplace=True)
        
        return result_df

    def _calculate_momentum_score(self, df: pd.DataFrame, trade_date: datetime.date) -> float:
        """计算动量分"""
        if df.empty or trade_date not in df.index:
            return 0.0
        
        # 计算均线
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema60'] = df['close'].ewm(span=60, adjust=False).mean()
        
        today = df.loc[trade_date]
        
        # 条件1: 价格高于关键均线
        price_above_ema20 = 1 if today['close'] > today['ema20'] else 0
        price_above_ema60 = 1 if today['close'] > today['ema60'] else 0
        
        # 条件2: 均线多头排列
        ema_bullish = 1 if today['ema20'] > today['ema60'] else 0
        
        # 条件3: 近期涨幅
        pct_change_5d = (today['close'] / df['close'].shift(5).loc[trade_date]) - 1 if len(df) > 5 else 0
        
        # 综合打分
        score = (price_above_ema20 * 2 + price_above_ema60 * 1 + ema_bullish * 2 + (pct_change_5d * 10))
        return score

    def _calculate_fund_flow_score(self, df: pd.DataFrame, trade_date: datetime.date) -> float:
        """计算资金流分"""
        if df.empty or trade_date not in df.index:
            return 0.0
            
        # 截取近期数据
        recent_df = df.loc[:trade_date].tail(self.fund_flow_lookback)
        if recent_df.empty:
            return 0.0
            
        # 条件1: 近期累计净流入
        net_inflow_sum = recent_df['net_amount'].sum()
        
        # 条件2: 净流入天数占比
        inflow_days_ratio = (recent_df['net_amount'] > 0).sum() / len(recent_df)
        
        # 综合打分
        score = (net_inflow_sum * 0.1 + inflow_days_ratio * 5)
        return score

    async def _calculate_breadth_score(self, industry_code: str, trade_date: datetime.date) -> float:
        """计算内部强度分"""
        members = await self.indicator_dao.get_industry_members(industry_code)
        if not members:
            return 0.0
            
        members_df = await self.indicator_dao.get_stocks_daily_close(members, trade_date)
        if members_df.empty:
            return 0.0
            
        # 计算上涨家数占比
        members_df['is_up'] = members_df['close'] > members_df['pre_close']
        up_ratio = members_df['is_up'].sum() / len(members_df)
        
        # 综合打分
        score = up_ratio * 10
        return score

    async def _calculate_leader_score(self, industry_code: str, trade_date: datetime.date) -> float:
        """
        【新增】计算龙头效应得分。
        简易版：直接使用数据源提供的领涨股。
        如果领涨股当天涨幅 > 5%，则认为龙头效应强。
        """
        # 假设 indicator_dao 有方法可以获取最新的行业资金流数据
        latest_flow = await self.indicator_dao.get_latest_industry_fund_flow(industry_code, trade_date)
        if latest_flow and latest_flow.pct_change_stock and latest_flow.pct_change_stock > 5.0:
            # print(f"      - [龙头效应] 发现领涨股 {latest_flow.lead_stock} 大涨 ({latest_flow.pct_change_stock}%)，得分: 1.0")
            return 1.0 # 强龙头效应
        elif latest_flow and latest_flow.pct_change_stock:
            # print(f"      - [龙头效应] 领涨股 {latest_flow.lead_stock} 涨幅 ({latest_flow.pct_change_stock}%) 未达阈值，得分: 0.5")
            return 0.5 # 弱龙头效应
        # print(f"      - [龙头效应] 未找到领涨股数据，得分: 0.0")
        return 0.0 # 无龙头效应

    async def _calculate_cohesion_score(self, industry_code: str, trade_date: datetime.date) -> float:
        """
        【新增】计算板块协同性（上涨广度）得分。
        统计板块内上涨家数占比和涨幅超过5%的家数。
        """
        # 1. 获取板块成分股
        members = await self.indicator_dao.get_industry_members(industry_code)
        if not members:
            return 0.0
        
        member_codes = [m.stock_id for m in members]
        
        # 2. 批量获取成分股当日行情
        # 假设 indicator_dao 有方法可以批量获取多只股票的单日行情
        daily_data = await self.stock_trade_dao.get_stocks_daily_data(member_codes, trade_date)
        if not daily_data:
            return 0.0

        total_count = len(members)
        rising_count = sum(1 for d in daily_data if d.pct_change > 0)
        strong_rising_count = sum(1 for d in daily_data if d.pct_change > 5.0)
        
        rising_ratio = rising_count / total_count if total_count > 0 else 0
        
        score = 0.0
        if rising_ratio > 0.7 and strong_rising_count >= 2:
            score = 1.0 # 极强协同性：普涨且有攻击梯队
        elif rising_ratio > 0.5:
            score = 0.6 # 较强协同性：大部分上涨
        elif rising_ratio > 0.3:
            score = 0.2 # 弱协同性：部分上涨
            
        print(f"      - [协同性] 上涨家数/总数: {rising_count}/{total_count} (占比: {rising_ratio:.2%}), 大涨家数: {strong_rising_count}，得分: {score}")
        return score

    async def _calculate_limit_up_echelon_score(self, industry_code: str, trade_date: datetime.date) -> float:
        """
        【新增-核心】计算涨停梯队得分。
        这是A股市场最强的信号之一。
        """
        # 1. 获取板块成分股
        members = await self.indicator_dao.get_industry_members(industry_code)
        if not members:
            return 0.0
            
        member_codes = [m.stock_id for m in members]
        
        # 2. 批量获取成分股当日基本面数据（包含涨停状态）
        # 假设 indicator_dao 有方法可以批量获取多只股票的单日基本面
        daily_basics = await self.indicator_dao.get_stocks_daily_basic(member_codes, trade_date)
        if not daily_basics:
            return 0.0

        # 假设 limit_status=1 为涨停
        limit_up_count = sum(1 for b in daily_basics if b.limit_status == 1)
        
        score = 0.0
        if limit_up_count >= 5:
            score = 1.0 # 板块高潮
        elif limit_up_count >= 3:
            score = 0.8 # 梯队形成
        elif limit_up_count >= 1:
            score = 0.4 # 有涨停股，热点发酵
            
        print(f"      - [涨停梯队] 发现 {limit_up_count} 家涨停，得分: {score}")
        return score

    async def _calculate_relative_strength_score(self, industry_daily_df: pd.DataFrame, market_daily_df: pd.DataFrame) -> float:
        """
        【新增】计算行业相对大盘的强度得分。
        """
        if industry_daily_df.empty or market_daily_df.empty:
            return 0.0

        # 合并行业与大盘数据
        df = pd.merge(industry_daily_df[['close']], market_daily_df, left_index=True, right_index=True, how='inner')
        if df.empty:
            return 0.0
            
        # 1. 计算RS（相对强度）曲线
        df['rs'] = df['close'] / df['market_close']
        
        # 2. 计算RS的20日均线，用于判断短期趋势
        df['rs_ma20'] = df['rs'].rolling(20).mean()
        
        # 3. 评分逻辑
        latest = df.iloc[-1]
        score = 0.0
        # 如果RS值在均线之上，说明短期强势
        if latest['rs'] > latest['rs_ma20']:
            score = 1.0
        
        # print(f"      - [相对强度] RS值: {latest['rs']:.2f}, RS_MA20: {latest['rs_ma20']:.2f}, 得分: {score:.2f}")
        return score

    async def analyze_industry_rotation(self, end_date: datetime.date, lookback_days: int = 10, market_code: str = '000300.SH') -> pd.DataFrame:
        """
        【新增】分析行业轮动，识别强度排名持续上升的板块。
        这是一个高阶扫描器，用于发现潜在的市场新主线。
        """
        print(f"\n--- [行业轮动分析] 开始分析截至 {end_date} 的过去 {lookback_days} 天行业轮动情况 ---")
        
        # 1. 获取过去N个交易日的日期列表 (需要一个交易日历工具)
        # 此处简化处理，实际应从交易日历服务获取
        trade_dates = [end_date - datetime.timedelta(days=i) for i in range(lookback_days)]
        
        # 2. 并行计算每一天的行业排名
        tasks = [self.calculate_industry_strength_rank(td, market_code) for td in trade_dates]
        daily_rank_results = await asyncio.gather(*tasks)
        
        # 3. 合并所有日期的排名数据
        all_ranks = []
        for i, df_rank in enumerate(daily_rank_results):
            if not df_rank.empty:
                df_rank['trade_date'] = trade_dates[i]
                all_ranks.append(df_rank.reset_index())
        
        if not all_ranks:
            print("    - [行业轮动分析] 未能获取任何历史排名数据，分析中止。")
            return pd.DataFrame()
            
        rotation_df = pd.concat(all_ranks, ignore_index=True)
        
        # 4. 为每个行业计算其排名的时间序列趋势
        def calculate_rank_momentum(group):
            # 确保数据按时间升序排列
            group = group.sort_values('trade_date')
            if len(group) < 3: # 数据点太少，无法计算趋势
                return pd.Series({'rank_momentum': 0, 'latest_rank': group['strength_rank'].iloc[-1]})
            
            # 使用线性回归计算斜率来代表动量
            # x轴是时间（天数），y轴是排名
            x = np.arange(len(group))
            y = group['strength_rank'].values
            slope, _ = np.polyfit(x, y, 1)
            
            return pd.Series({'rank_momentum': slope, 'latest_rank': y[-1]})

        # 按行业分组，计算动量
        rotation_momentum = rotation_df.groupby('industry_code').apply(calculate_rank_momentum)
        
        # 合并行业名称
        industry_names = rotation_df[['industry_code', 'industry_name']].drop_duplicates().set_index('industry_code')
        final_report = pd.merge(rotation_momentum, industry_names, left_index=True, right_index=True)
        
        print("--- [行业轮动分析] 分析完成 ---")
        # 按动量降序排序，动量为正且越大，说明排名上升越快
        return final_report.sort_values('rank_momentum', ascending=False)


    # async def prepare_data_with_industry_context(
    #     self,
    #     stock_code: str,
    #     config: dict,
    #     trade_time: Optional[str] = None
    # ) -> Dict[str, pd.DataFrame]:
    #     """
    #     【V6.0 行业背景版】准备数据，并自动注入行业强度上下文。
    #     这是策略总指挥应该调用的主要入口。

    #     Args:
    #         stock_code (str): 股票代码。
    #         config (dict): 策略配置文件字典。
    #         trade_time (Optional[str]): 交易时间。

    #     Returns:
    #         Dict[str, pd.DataFrame]: 包含所有时间周期DataFrame的字典，
    #                                  其中日线数据已注入 'industry_strength_rank_D' 列。
    #     """
    #     print(f"--- [数据准备V6.0-行业版] 开始为 {stock_code} 准备数据及行业背景... ---")
        
    #     # --- 步骤 1: 调用现有的 prepare_data 获取基础K线和指标数据 ---
    #     all_dfs = await self.prepare_data(stock_code, config, trade_time)

    #     if not all_dfs or 'D' not in all_dfs or all_dfs['D'].empty:
    #         logger.warning(f"[{stock_code}] 基础数据准备失败，无法注入行业背景。")
    #         return all_dfs # 返回空字典或不完整的字典

    #     # --- 步骤 2: 获取并注入行业强度数据 ---
    #     logger.info(f"    - [行业背景注入] 开始为 {stock_code} 获取行业强度...")
        
    #     # 解析交易日期
    #     current_trade_date = pd.to_datetime(trade_time).date() if trade_time else datetime.now().date()

    #     # 1. 计算所有行业的强度排名 (注：在批量执行时，此结果应由外部调用者缓存)
    #     #    这里假设调用者会处理缓存，Service只负责计算。
    #     industry_rank_df = await self.calculate_industry_strength_rank(current_trade_date)

    #     # 2. 查询当前股票所属的行业代码 (使用 self.stock_basic_dao)
    #     #    假设 StockBasicInfoDao 有 get_stock_industry_info 方法
    #     stock_industry_info = await self.indicator_dao.get_stock_industry_info(stock_code)
    #     stock_industry_code = stock_industry_info.get('code') if stock_industry_info else None
    #     stock_industry_name = stock_industry_info.get('name') if stock_industry_info else '未知行业'

    #     # 3. 从排名中找到该行业的强度排名
    #     stock_industry_rank = 0.0 # 默认为0 (无行业或未找到排名)
    #     if not industry_rank_df.empty and stock_industry_code and stock_industry_code in industry_rank_df.index:
    #         stock_industry_rank = industry_rank_df.loc[stock_industry_code, 'strength_rank']
    #         print(f"    - [行业背景注入] 股票 {stock_code} 所属行业 '{stock_industry_name}'({stock_industry_code}) 当日强度排名: {stock_industry_rank:.2%}")
    #     else:
    #         print(f"    - [行业背景注入] 股票 {stock_code} ({stock_industry_name}) 未找到行业排名，默认排名为 0.0。")

    #     # 4. 将行业强度排名作为一个新特征，合并到日线DataFrame中
    #     all_dfs['D']['industry_strength_rank_D'] = stock_industry_rank
    #     print(f"    - [行业背景注入] 已将 'industry_strength_rank_D' 列 ({stock_industry_rank:.2f}) 添加到日线数据。")
        
    #     print(f"--- [数据准备V6.0-行业版] {stock_code} 数据准备完成。 ---")
    #     return all_dfs


    # # 修改方法: 重构 prepare_multi_timeframe_data 以提高效率和正确性
    # async def prepare_multi_timeframe_data(
    #     self,
    #     stock_code: str,
    #     config_paths: List[str],
    #     trade_time: Optional[str] = None
    # ) -> Dict[str, pd.DataFrame]:
    #     """
    #     【V3.1 务实版】根据多个配置文件，智能合并需求，准备所有时间周期数据。
    #     - 核心逻辑:
    #       1. 遍历所有配置文件路径，读取并合并 'feature_engineering_params'。
    #       2. 从合并后的配置中，自动扫描并识别所有需要的时间周期 (e.g., D, W, 60, 15, 5)。
    #       3. 并行获取所有周期的原始K线数据。
    #       4. 为每个周期的数据独立计算其所需的指标（使用合并后的总配置）。
    #       5. 将周线/月线战略指标数据合并到日线数据中，并执行前向填充。
    #       6. 返回一个包含所有处理后DataFrame的字典。
    #     """
    #     print(f"    [调试-数据服务 V3.1] 开始为 {stock_code} 准备多时间框架数据...")
    #     print(f"    [调试-数据服务 V3.1] 使用的配置文件: {config_paths}")

    #     # --- 步骤 1: 读取并合并所有配置文件中的指标需求 ---
    #     merged_feature_params = {"indicators": {}}
        
    #     for path in config_paths:
    #         try:
    #             with open(path, 'r', encoding='utf-8') as f:
    #                 params = json.load(f)
                
    #             current_indicators = params.get('feature_engineering_params', {}).get('indicators', {})
                
    #             # 智能合并指标配置
    #             for key, value in current_indicators.items():
    #                 if key not in merged_feature_params["indicators"]:
    #                     merged_feature_params["indicators"][key] = deepcopy(value)
    #                 else:
    #                     # 如果指标已存在，合并 apply_on 列表，去重
    #                     if isinstance(value, dict) and 'apply_on' in value:
    #                         existing_apply_on = set(merged_feature_params["indicators"][key].get('apply_on', []))
    #                         new_apply_on = set(value.get('apply_on', []))
    #                         merged_feature_params["indicators"][key]['apply_on'] = sorted(list(existing_apply_on.union(new_apply_on)))

    #         except Exception as e:
    #             logger.error(f"[{stock_code}] 读取或解析配置文件 {path} 失败: {e}")
    #             return {}

    #     # --- 步骤 2: 从合并后的配置中识别所有需要的时间周期 ---
    #     required_tfs: Set[str] = set()
    #     for indicator_config in merged_feature_params["indicators"].values():
    #         if isinstance(indicator_config, dict) and indicator_config.get('enabled', False):
    #             apply_on = indicator_config.get('apply_on', [])
    #             for tf in apply_on:
    #                 required_tfs.add(str(tf))

    #     # 确保日线和周线作为基础周期存在
    #     required_tfs.add('D')
    #     required_tfs.add('W')
    #     print(f"    [调试-数据服务 V3.1] 合并配置后，识别出需要的数据周期: {sorted(list(required_tfs))}")

    #     # --- 步骤 3: 并行获取所有周期的原始K线数据 ---
    #     async def _fetch_raw_data(tf: str):
    #         print(f"      - 开始获取 {tf} 周期原始数据...")
    #         df = await self.get_data(stock_code, tf, trade_time=trade_time)
    #         if df is None or df.empty:
    #             logger.warning(f"[{stock_code}] 获取 {tf} 周期数据失败或为空。")
    #             return tf, None
    #         print(f"      - 成功获取 {tf} 周期原始数据 {len(df)} 条。")
    #         return tf, df

    #     tasks = [_fetch_raw_data(tf) for tf in required_tfs]
    #     raw_data_results = await asyncio.gather(*tasks, return_exceptions=True)

    #     raw_dfs: Dict[str, pd.DataFrame] = {}
    #     for res in raw_data_results:
    #         if isinstance(res, Exception) or res[1] is None:
    #             logger.error(f"[{stock_code}] 获取数据时发生错误: {res}")
    #             continue
    #         tf, df = res
    #         raw_dfs[tf] = df

    #     if 'D' not in raw_dfs or raw_dfs['D'].empty:
    #         logger.error(f"[{stock_code}] 核心的日线数据获取失败，无法继续分析。")
    #         return {}

    #     # --- 步骤 4: 为每个周期的数据独立计算指标 ---
    #     processed_dfs: Dict[str, pd.DataFrame] = {}
    #     for tf, df in raw_dfs.items():
    #         print(f"    [调试-数据服务 V3.1] 开始为 {tf} 周期数据计算技术指标...")
    #         # 使用合并后的总配置来计算指标
    #         df_with_indicators, _ = self.calculate_indicators_for_dataframe(df, merged_feature_params["indicators"], tf)
    #         processed_dfs[tf] = df_with_indicators
    #         print(f"      - {tf} 周期指标计算完成。")

    #     # --- 步骤 5: 将战略数据(周/月)指标合并到日线数据中 ---
    #     df_daily_final = processed_dfs['D'].copy()

    #     for strategic_tf in ['W', 'M']:
    #         if strategic_tf in processed_dfs:
    #             df_strategic = processed_dfs[strategic_tf]
    #             # 仅选择指标列，避免合并OHLC等基础列
    #             strategic_indicator_cols = [col for col in df_strategic.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'amount']]
    #             df_strategic_indicators = df_strategic[strategic_indicator_cols]
                
    #             # 添加后缀以便区分
    #             df_strategic_indicators = df_strategic_indicators.add_suffix(f'_{strategic_tf}')
                
    #             # 合并
    #             df_daily_final = pd.merge(df_daily_final, df_strategic_indicators, left_index=True, right_index=True, how='left')
    #             print(f"    [调试-数据服务 V3.1] 已将 {strategic_tf} 周期指标数据合并到日线数据。")

    #             # 前向填充
    #             strategic_cols_to_fill = [col for col in df_daily_final.columns if col.endswith(f'_{strategic_tf}')]
    #             if strategic_cols_to_fill:
    #                 df_daily_final[strategic_cols_to_fill] = df_daily_final[strategic_cols_to_fill].ffill()
    #                 for col in strategic_cols_to_fill:
    #                     # 检查列是否存在，因为ffill后可能全为NaN而被删除
    #                     if col in df_daily_final.columns:
    #                         if pd.api.types.is_bool_dtype(df_daily_final[col]):
    #                             df_daily_final[col].fillna(False, inplace=True)
    #                         elif pd.api.types.is_numeric_dtype(df_daily_final[col]):
    #                             df_daily_final[col].fillna(0, inplace=True)
    #                 print(f"      - 已对 {len(strategic_cols_to_fill)} 个 {strategic_tf} 周期列执行前向填充(ffill)。")

    #     # --- 步骤 6: 组装最终的返回字典 ---
    #     final_dfs: Dict[str, pd.DataFrame] = {}
    #     # 将处理好的、带有战略背景的日线数据放入字典
    #     final_dfs['D'] = df_daily_final

    #     # 将其他分钟级别的数据放入字典
    #     for tf in required_tfs:
    #         # 不再需要单独放入 W 和 M，因为它们的信息已经在 D 中了
    #         if tf not in ['D', 'W', 'M'] and tf in processed_dfs:
    #             final_dfs[tf] = processed_dfs[tf]

    #     print(f"    [调试-数据服务 V3.1] 数据准备完成，最终字典包含的周期: {list(final_dfs.keys())}")
    #     return final_dfs

    # async def prepare_daily_centric_dataframe(
    #     self,
    #     stock_code: str,
    #     trade_time: str,
    #     daily_config_path: Optional[str] = None,
    #     weekly_config_path: Optional[str] = None,
    #     monthly_config_path: Optional[str] = None
    # ) -> Optional[pd.DataFrame]:
    #     """
    #     【V2.2 适配V5.0配置】准备以日线为中心，融合了周线/月线指标的DataFrame。
    #     - 此方法本身无需修改，因为它调用的 _calculate_indicators_for_timescale 已升级。
    #     """
    #     # 1. 加载所有需要的配置
    #     # (此部分代码与您提供的版本完全一致，保持不变)
    #     daily_config = self._load_config(daily_config_path) if daily_config_path else {}
    #     weekly_config = self._load_config(weekly_config_path) if weekly_config_path else {}
    #     monthly_config = self._load_config(monthly_config_path) if monthly_config_path else {}
        
    #     daily_indicators = daily_config.get('feature_engineering_params', {}).get('indicators', {})
    #     weekly_indicators = weekly_config.get('feature_engineering_params', {}).get('indicators', {})
    #     monthly_indicators = monthly_config.get('feature_engineering_params', {}).get('indicators', {})
    #     print(f"--- [数据准备V5.0] 开始为 {stock_code} 构建多时间框架数据 ---")

    #     # 2. 获取基础日线数据及资金流数据
    #     # (此部分代码与您提供的版本完全一致，保持不变)
    #     df_daily = await self._get_ohlcv_data(stock_code, 'D', 500, trade_time)
    #     if df_daily is None or df_daily.empty:
    #         logger.error(f"[{stock_code}] 获取日线基础数据失败。")
    #         return None
        
    #     # 步骤 2.5: 获取并合并资金流和筹码数据
    #     print(f"--- [数据准备V2.3] 正在调用整合DAO获取资金流和筹码信息... ---")
        
    #     # 将字符串格式的 trade_time 转换为 datetime 对象，以匹配DAO方法的类型提示
    #     trade_time_dt = pd.to_datetime(trade_time) if trade_time else None

    #     # 单次调用新的DAO方法
    #     df_fund_chips = await self.strategies_dao.get_fund_flow_and_chips_data(
    #         stock_code=stock_code,
    #         trade_time=trade_time_dt
    #     )

    #     # 将获取到的整合数据合并到日线DataFrame中
    #     if df_fund_chips is not None and not df_fund_chips.empty:
    #         if df_daily.index.tz is not None:
    #             print(f"    - [时区统一] 检测到 df_daily 索引带有时区 ({df_daily.index.tz})，正在移除...")
    #             df_daily.index = df_daily.index.tz_localize(None)
            
    #         if df_fund_chips.index.tz is not None:
    #             print(f"    - [时区统一] 检测到 df_fund_chips 索引带有时区 ({df_fund_chips.index.tz})，正在移除...")
    #             df_fund_chips.index = df_fund_chips.index.tz_localize(None)
    #         # 使用左连接将资金筹码数据合并到日线数据上
    #         df_daily = pd.merge(df_daily, df_fund_chips, left_index=True, right_index=True, how='left')
    #         print("    - [信息] 已成功合并资金流与筹码的整合数据。")
            
    #         # 对新合并的列中因左连接产生的NaN值进行填充
    #         # DAO内部的ffill处理了数据内部的缺失，这里的fillna(0)处理因对齐产生的头部缺失
    #         new_cols = list(df_fund_chips.columns)
    #         df_daily[new_cols] = df_daily[new_cols].fillna(0)
    #         print(f"    - [信息] 已对 {len(new_cols)} 个新增资金/筹码列的NaN值填充为0。")
    #     else:
    #         print("    - [警告] 未能获取到资金流和筹码数据。")

    #     # 3. 计算日线指标
    #     # (独立计算) 调用通用计算器，传入日线数据、日线配置和 "_D" 后缀
    #     df_daily = await self._calculate_indicators_for_timescale(df_daily, daily_indicators, 'D')

    #     # 4. 聚合为周线
    #     # (数据转换) 将日线数据聚合为周线
    #     if weekly_indicators:
    #         df_weekly = self._resample_to_weekly(df_daily.copy())
    #         df_weekly = await self._calculate_indicators_for_timescale(df_weekly, weekly_indicators, 'W')
            
    #         df_daily.sort_index(inplace=True)
    #         df_weekly.sort_index(inplace=True)
    #         df_daily = pd.merge_asof(df_daily, df_weekly, left_index=True, right_index=True, direction='backward')
    #         print(f"    - [数据准备-战略层] 已成功合并周线指标。")

    #     # 5. 准备并合并月线数据 (未来扩展点)
    #     if monthly_indicators:
    #         df_monthly = self._resample_to_monthly(df_daily.copy())
    #         df_monthly = await self._calculate_indicators_for_timescale(df_monthly, monthly_indicators, 'M')
            
    #         df_daily.sort_index(inplace=True)
    #         df_monthly.sort_index(inplace=True)
    #         df_daily = pd.merge_asof(df_daily, df_monthly, left_index=True, right_index=True, direction='backward')
    #         print(f"    - [数据准备-战略层] 已成功合并月线指标。")

    #     # 6. 数据清洗和返回
    #     # (此部分代码与您提供的版本完全一致，保持不变)
    #     reliable_col = next((col for col in df_daily.columns if col.endswith('_W')), None)
    #     if reliable_col:
    #         df_daily.dropna(subset=[reliable_col], inplace=True)
        
    #     print(f"--- [数据准备V5.0] 数据准备完成 ---")
    #     return df_daily

    # async def prepare_minute_centric_dataframe(self, stock_code: str, params_file: str, timeframe: str, trade_time: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    #     """
    #     【V2.1 适配V5.0配置】为指定的分钟线周期准备DataFrame。
    #     - 此方法本身无需修改，因为它调用的 _calculate_indicators_for_timescale 已升级。
    #     """
    #     print(f"--- [数据准备-分钟线] 开始为 {stock_code} ({timeframe}周期) 准备数据 ---")
    #     try:
    #         params = self._load_config(params_file)
    #         fe_params = params.get('feature_engineering_params', {})
    #         indicators_to_calc = fe_params.get('indicators', {})
            
    #         needed_bars = fe_params.get('base_needed_bars', 1000)
    #         df_minute = await self._get_ohlcv_data(stock_code, timeframe, needed_bars, trade_time)
            
    #         if df_minute is None or df_minute.empty:
    #             logger.warning(f"[{stock_code}] 无法获取周期 '{timeframe}' 的数据，跳过。")
    #             return None, None
            
    #         print(f"    - [数据准备-分钟线] 成功获取 {len(df_minute)} 条 {timeframe} 周期K线。")

    #         # 【核心】调用已升级的通用指标计算器
    #         df_final = await self._calculate_indicators_for_timescale(
    #             df=df_minute, 
    #             config=indicators_to_calc, 
    #             timeframe_key=timeframe # 传递分钟线标识，如 '60', '15'
    #         )
            
    #         print(f"    - [数据准备-分钟线] 指标计算完成，最终DataFrame包含列: {df_final.columns.to_list()}")
    #         return df_final, params

    #     except FileNotFoundError:
    #         logger.error(f"[{stock_code}] 分钟线配置文件未找到: {params_file}")
    #         return None, None
    #     except Exception as e:
    #         logger.error(f"[{stock_code}] 为周期 '{timeframe}' 准备数据时出错: {e}", exc_info=True)
    #         traceback.print_exc()
    #         return None, None

    # def _resample_to_weekly(self, df_daily: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     【V2.0 修正版】将日线数据聚合为周线数据。
    #     - 修正：在聚合后，为所有列统一添加 '_W' 后缀，确保数据一致性。
    #     - 优化：使用 dropna(how='all')，仅在某一周完全没有数据时才删除，更加健壮。
    #     """
    #     print("  [步骤3] 正在将日线聚合为周线...")
    #     # 'W-FRI' 表示以周五为每周的结束点，更符合A股交易习惯
    #     ohlc_dict = {
    #         'open': 'first', 
    #         'high': 'max', 
    #         'low': 'min', 
    #         'close': 'last', 
    #         'volume': 'sum'
    #     }
    #     df_weekly = df_daily.resample('W-FRI').agg(ohlc_dict)

    #     # 1. 为所有聚合后的列添加 '_W' 后缀，实现命名统一
    #     df_weekly.columns = [f"{col}_W" for col in df_weekly.columns]
    #     print(f"    - 已将周线基础列重命名为: {list(df_weekly.columns)}") # 增加一条调试信息

    #     # 2. 优化dropna逻辑，只删除所有数据都为NaN的行（例如国庆长假所在的周）
    #     df_weekly.dropna(how='all', inplace=True)

    #     return df_weekly


    # logger.info(f"[{stock_code}] 开始补充外部特征 (指数、板块、筹码、资金流向)...")
    # final_df = await self.enrich_features(df=final_df, stock_code=stock_code, main_indices=main_index_codes, external_data_history_days=external_data_history_days)
    # logger.info(f"[{stock_code}] 外部特征补充完成。最终 DataFrame Shape: {final_df.shape}, 列数: {len(final_df.columns)}")
    # actual_rsi_period = bs_params.get('rsi_period', default_rsi_p['period'])
    # actual_macd_fast = bs_params.get('macd_fast', default_macd_p['period_fast'])
    # actual_macd_slow = bs_params.get('macd_slow', default_macd_p['period_slow'])
    # actual_macd_signal = bs_params.get('macd_signal', default_macd_p['signal_period'])
    # fe_config = params.get('feature_engineering_params', {})
    # apply_on_tfs = fe_config.get('apply_on_timeframes', bs_timeframes)

    # # 相对强度
    # rs_config = fe_config.get('relative_strength', {})
    # if rs_config.get('enabled', False):
    #      ths_indexs_for_rs_objects = await self.industry_dao.get_stock_ths_indices(stock_code)
    #      if ths_indexs_for_rs_objects is None:
    #           logger.warning(f"[{stock_code}] 无法获取股票 {stock_code} 的同花顺板块信息。相对强度计算将跳过。")
    #           ths_codes_for_rs = []
    #      else:
    #         ths_codes_for_rs = [m.ths_index.ts_code for m in ths_indexs_for_rs_objects if m.ths_index]
    #      all_benchmark_codes_for_rs = list(set(main_index_codes + ths_codes_for_rs))
    #      periods = rs_config.get('periods', [5, 10, 20])
    #      if all_benchmark_codes_for_rs and periods:
    #           for tf_apply in apply_on_tfs:
    #                stock_close_col = f'close_{tf_apply}'
    #                if stock_close_col in final_df.columns:
    #                     final_df = self.calculate_relative_strength(df=final_df, stock_close_col=stock_close_col, benchmark_codes=all_benchmark_codes_for_rs, periods=periods, time_level=tf_apply)
    #                else:
    #                     logger.warning(f"[{stock_code}] 计算相对强度 for TF {tf_apply} 失败，未找到股票收盘价列: {stock_close_col}")
    #           logger.info(f"[{stock_code}] 相对强度/超额收益特征计算完成。")
    #      else:
    #           logger.warning(f"[{stock_code}] 相对强度/超额收益特征未启用或配置不完整 (基准代码或周期列表为空)。")
    # # 滞后特征
    # lag_config = fe_config.get('lagged_features', {})
    # if lag_config.get('enabled', False):
    #     columns_to_lag_from_json = lag_config.get('columns_to_lag', [])
    #     lags = lag_config.get('lags', [1, 2, 3])
    #     if columns_to_lag_from_json and lags:
    #         logger.debug(f"[{stock_code}] 开始添加滞后特征...")
    #         for tf_apply in apply_on_tfs:
    #             actual_cols_for_lagging = []
    #             for col_template_from_json in columns_to_lag_from_json:
    #                 effective_base_name = col_template_from_json
    #                 if col_template_from_json.startswith("RSI"):
    #                     effective_base_name = f"RSI_{actual_rsi_period}"
    #                 elif col_template_from_json.startswith("MACD_") and \
    #                     not col_template_from_json.startswith("MACDh_") and \
    #                     not col_template_from_json.startswith("MACDs_"):
    #                     effective_base_name = f"MACD_{actual_macd_fast}_{actual_macd_slow}_{actual_macd_signal}"
    #                 elif col_template_from_json.startswith("MACDh_"):
    #                     effective_base_name = f"MACDh_{actual_macd_fast}_{actual_macd_slow}_{actual_macd_signal}"
    #                 elif col_template_from_json.startswith("MACDs_"):
    #                     effective_base_name = f"MACDs_{actual_macd_fast}_{actual_macd_slow}_{actual_macd_signal}"
    #                 col_with_suffix = f"{effective_base_name}_{tf_apply}"
    #                 if col_with_suffix in final_df.columns:
    #                     actual_cols_for_lagging.append(col_with_suffix)
    #                 else:
    #                     if effective_base_name in final_df.columns:
    #                         actual_cols_for_lagging.append(effective_base_name)
    #                         logger.debug(f"[{stock_code}] TF {tf_apply}: 找到不带时间级别后缀的列 {effective_base_name} (来自JSON模板 {col_template_from_json}) 进行滞后计算。")
    #                     else:
    #                         logger.warning(f"[{stock_code}] TF {tf_apply}: 未找到指定列 {effective_base_name} (来自JSON模板 {col_template_from_json}) 或其带后缀形式 {col_with_suffix} 进行滞后计算。")
    #             if actual_cols_for_lagging:
    #                 logger.debug(f"[{stock_code}] TF {tf_apply}: 准备为以下列添加滞后特征: {actual_cols_for_lagging}")
    #                 final_df = self.add_lagged_features(final_df, actual_cols_for_lagging, lags)
    #                 logger.debug(f"[{stock_code}] 添加滞后特征 for TF {tf_apply} 完成。")
    #             else:
    #                 logger.warning(f"[{stock_code}] 添加滞后特征 for TF {tf_apply} 失败，未找到任何有效列。JSON配置: {columns_to_lag_from_json}")
    #         logger.info(f"[{stock_code}] 滞后特征添加完成。")
    #     else:
    #         logger.warning(f"[{stock_code}] 滞后特征未启用或配置不完整。")
    # # 滚动特征
    # roll_config = fe_config.get('rolling_features', {})
    # if roll_config.get('enabled', False):
    #     columns_to_roll_from_json = roll_config.get('columns_to_roll', [])
    #     windows = roll_config.get('windows', [5, 10, 20])
    #     stats = roll_config.get('stats', ["mean", "std"])
    #     if columns_to_roll_from_json and windows and stats:
    #         logger.debug(f"[{stock_code}] 开始添加滚动统计特征...")
    #         for tf_apply in apply_on_tfs:
    #             actual_cols_for_rolling = []
    #             for col_template_from_json in columns_to_roll_from_json:
    #                 effective_base_name = col_template_from_json
    #                 if col_template_from_json.startswith("RSI"):
    #                     effective_base_name = f"RSI_{actual_rsi_period}"
    #                 elif col_template_from_json.startswith("MACD_") and \
    #                     not col_template_from_json.startswith("MACDh_") and \
    #                     not col_template_from_json.startswith("MACDs_"):
    #                     effective_base_name = f"MACD_{actual_macd_fast}_{actual_macd_slow}_{actual_macd_signal}"
    #                 elif col_template_from_json.startswith("MACDh_"):
    #                     effective_base_name = f"MACDh_{actual_macd_fast}_{actual_macd_slow}_{actual_macd_signal}"
    #                 elif col_template_from_json.startswith("MACDs_"):
    #                     effective_base_name = f"MACDs_{actual_macd_fast}_{actual_macd_slow}_{actual_macd_signal}"
    #                 col_with_suffix = f"{effective_base_name}_{tf_apply}"
    #                 if col_with_suffix in final_df.columns:
    #                     actual_cols_for_rolling.append(col_with_suffix)
    #                 else:
    #                     if effective_base_name in final_df.columns:
    #                         actual_cols_for_rolling.append(effective_base_name)
    #                         logger.debug(f"[{stock_code}] TF {tf_apply}: 找到不带时间级别后缀的列 {effective_base_name} (来自JSON模板 {col_template_from_json}) 进行滚动统计计算。")
    #                     else:
    #                         logger.warning(f"[{stock_code}] TF {tf_apply}: 未找到指定列 {effective_base_name} (来自JSON模板 {col_template_from_json}) 或其带后缀形式 {col_with_suffix} 进行滚动统计计算。")
    #             if actual_cols_for_rolling:
    #                 logger.debug(f"[{stock_code}] TF {tf_apply}: 准备为以下列添加滚动统计特征: {actual_cols_for_rolling}")
    #                 final_df = self.add_rolling_features(final_df, actual_cols_for_rolling, windows, stats)
    #                 logger.debug(f"[{stock_code}] 添加滚动统计特征 for TF {tf_apply} 完成。")
    #             else:
    #                 logger.warning(f"[{stock_code}] 滚动统计特征 for TF {tf_apply} 失败，未找到任何指定列。JSON配置: {columns_to_roll_from_json}")
    #         logger.info(f"[{stock_code}] 滚动统计特征添加完成。")
    #     else:
    #         logger.warning(f"[{stock_code}] 滚动统计特征未启用或配置不完整。")
    # original_nan_count = final_df.isnull().sum().sum()














