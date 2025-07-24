# services\indicator_services.py
import asyncio
import datetime
import json
import pytz
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
    【V8.0 情报锻造中心版】技术指标计算服务
    - 核心升级: 彻底重构周/月线数据的处理方式。不再独立获取预聚合的周/月线数据，
                而是强制从日线数据(D)中，通过 resample 方法，亲手锻造出高保真的
                周线(W)和月线(M)的OHLCV及核心指标。
    - 新增功能: 引入 _calculate_synthetic_weekly_indicators 辅助函数，专门负责
                计算像CMF这类必须依赖日线过程的复杂周线指标，确保情报的绝对准确性。
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

    def _log_final_data_columns(self, all_dfs: Dict[str, pd.DataFrame]):
        """
        【V225.0 新增】军械库清单生成器
        - 核心职责: 在所有数据准备和计算流程结束后，按周期清晰地打印出最终生成的所有数据列名。
        """
        print("\n" + "="*30 + " [最终军械库清单] " + "="*30)
        print("情报锻造中心已完成所有数据准备，最终产出的数据列清单如下：")

        # 定义周期的标准排序，确保输出顺序固定
        sorted_timeframes = sorted(
            all_dfs.keys(), 
            key=lambda x: (
                {'M': 0, 'W': 1, 'D': 2}.get(x, 3), # 月、周、日优先
                -int(x) if x.isdigit() else 0 # 分钟线按从大到小排序
            )
        )

        for timeframe in sorted_timeframes:
            df = all_dfs[timeframe]
            if df is None or df.empty:
                continue
            
            print(f"\n--- 周期: {timeframe} (共 {len(df.columns)} 列) ---")
            # 为了美观，每5个列名换一行打印
            columns_list = sorted(df.columns.tolist())
            for i in range(0, len(columns_list), 5):
                print("  ".join(f"{col:<30}" for col in columns_list[i:i+5]))

        print("\n" + "="*32 + " [清单生成完毕] " + "="*32 + "\n")

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
        # ▼▼▼【代码修改 V117.28】: 确保获取实时数据 ▼▼▼
        # 如果 trade_time 未提供（例如在实时触发的场景），则使用当前时间作为查询终点。
        # 这确保了DAO层能够获取到截至目前的最新数据，包括当天的盘中K线。
        df = await self.indicator_dao.get_history_ohlcv_df(
            stock_code=stock_code, 
            time_level=time_level, 
            limit=needed_bars, 
            trade_time=trade_time # 直接传递，不做任何处理
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
                df[time_col] = pd.to_datetime(df[time_col], utc=True)
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

    def _get_max_lookback_period(self, config: dict) -> int:
        """
        【军需官】扫描整个策略配置，找出所有指标中要求的最长回溯期。
        这是一个简化的实现，用于演示核心思想。
        """
        print("    - [军需官] 正在扫描全军军火库，确定最大回溯需求...")
        # 简化实现：
        calculated_max = 350 # 保守估计，足以满足EMA(55周)等大周期指标
        print(f"    - [军需官] 扫描完成，最大回溯需求估算为 {calculated_max} 个日线周期。")
        return calculated_max

    async def prepare_data_for_strategy(
        self,
        stock_code: str,
        config: dict,
        trade_time: Optional[str] = None,
        latest_only: bool = False  # <--- 安装新的“电报机”
    ) -> Dict[str, pd.DataFrame]:
        """
        【V8.0 情报锻造中心版】为策略准备数据的统一入口。
        此版本将调用重构后的核心数据准备函数。
        """
        # print(f"--- [数据准备V8.0-情报锻造中心] 开始为 {stock_code} 准备数据... ---")

        # --- 步骤 1: 调用重构后的核心数据准备函数 ---
        all_dfs = await self._prepare_base_data_and_indicators(stock_code, config, trade_time, latest_only=latest_only)

        # --- 后续逻辑保持不变，注入行业背景、游资信号等 ---
        if not all_dfs or 'D' not in all_dfs or all_dfs['D'].empty:
            # logger.warning(f"[{stock_code}] 基础数据准备失败，无法继续。")
            return all_dfs
        
        df_daily = all_dfs['D']
        start_date = df_daily.index.min().date()
        end_date = df_daily.index.max().date()

        industry_params = self._find_params_recursively(config, 'industry_context_params')
        is_industry_enabled = industry_params.get('enabled', False) if industry_params else False
        
        hot_money_params = self._find_params_recursively(config, 'hot_money_params')
        is_hm_enabled = hot_money_params.get('enabled', False) if hot_money_params else False

        if is_hm_enabled:
            # print("    - [配置信息] 检测到游资分析已启用，开始获取并处理游资信号...")
            hm_signals_df = await self._prepare_hot_money_signals(stock_code, start_date, end_date, hot_money_params)
            if not hm_signals_df.empty:
                df_daily = df_daily.merge(hm_signals_df, left_index=True, right_index=True, how='left')
                for col in hm_signals_df.columns:
                    if col in df_daily.columns:
                        df_daily[col] = df_daily[col].fillna(False).astype(bool)
                # print(f"    - [游资信号注入] 已将游资原子信号列注入日线数据。")
        
        all_dfs['D'] = df_daily

        if is_industry_enabled:
            # print(f"    - [配置信息] 检测到行业协同已启用，开始获取行业强度...")
            current_trade_date = pd.to_datetime(trade_time, utc=True).date() if trade_time else datetime.datetime.now().date()
            industry_rank_df = await self.calculate_industry_strength_rank(current_trade_date)
            stock_industry_info = await self.indicator_dao.get_stock_industry_info(stock_code)
            stock_industry_code = stock_industry_info.get('code') if stock_industry_info else None
            stock_industry_name = stock_industry_info.get('name') if stock_industry_info else '未知行业'
            stock_industry_rank = 0.0
            if not industry_rank_df.empty and stock_industry_code and stock_industry_code in industry_rank_df.index:
                stock_industry_rank = industry_rank_df.loc[stock_industry_code, 'strength_rank']
            all_dfs['D']['industry_strength_rank_D'] = stock_industry_rank
            # print(f"    - [行业背景注入] 已将 'industry_strength_rank_D' 列注入日线数据。")

        # 在所有基础指标计算完毕后，调用斜率计算
        all_dfs = await self._calculate_all_slopes(all_dfs, config)
        
        #  调用军械库清单生成器 ▼▼▼
        # self._log_final_data_columns(all_dfs)
        
        return all_dfs

    async def _prepare_base_data_and_indicators(
        self,
        stock_code: str,
        config: dict,
        trade_time: Optional[str] = None,
        latest_only: bool = False # <--- 接收并执行指令
    ) -> Dict[str, pd.DataFrame]:
        """
        【V8.1 情报锻造中心-完整版】
        - 核心重构: 不再独立获取周/月线数据，而是从日线数据中通过 resample 合成。
        - 新增功能: 调用 _calculate_synthetic_weekly_indicators 合成复杂的周线指标。
        - 流程整合: 完整集成了旧筹码、新筹码等补充数据的并发获取与合并逻辑。
        - 日志优化: 提供了更清晰的数据处理流程追踪日志。
        """
        print(f"--- [数据准备V8.1] 开始为 {stock_code} 准备基础数据与指标 ---")
        
        # 1. 从配置中解析需要哪些时间周期的数据
        required_tfs = self._discover_required_timeframes_from_config(config)
        if not required_tfs:
            print("    - [配置读取] 未发现任何需要的时间周期，处理终止。")
            return {}
        
        # ▼▼▼ 闪电模式的核心实现 ▼▼▼
        # 步骤1: 根据模式决定基础数据量
        if latest_only:
            # 闪电模式：智能计算最少需要的数据量
            max_lookback = self._get_max_lookback_period(config)
            safety_buffer = 100 # 增加一个慷慨的安全缓冲,确保周线合成和指标预热
            base_needed_bars = max_lookback + safety_buffer
            print(f"    - [闪电模式启动] 策略最大回溯期: {max_lookback}, 安全缓冲: {safety_buffer}, 最终加载: {base_needed_bars} 条记录。")
        else:
            # 全面模式：使用配置中的默认值
            base_needed_bars = config.get('feature_engineering_params', {}).get('base_needed_bars', 1200)
        
        base_needed_bars = config.get('feature_engineering_params', {}).get('base_needed_bars', 500)
        # print(f"    - [配置读取] 策略请求的基础数据量: {base_needed_bars} bars, 需要的周期: {sorted(list(required_tfs))}")

        # 2. 【核心修改】确定需要从API获取的“基础”时间周期
        base_tfs_to_fetch = set()
        resample_map = {} # 记录需要从哪个源周期合成目标周期
        for tf in required_tfs:
            if tf in ['W', 'M']:
                base_tfs_to_fetch.add('D') # 周线和月线都需要日线作为原材料
                resample_map[tf] = 'D'
            else:
                base_tfs_to_fetch.add(tf) # 其他周期（如D, 60, 30）直接获取

        # 3. 检查并准备所有补充数据的获取任务
        indicators_config = config.get('feature_engineering_params', {}).get('indicators', {})
        tasks = []

        # 任务准备: 旧筹码和资金流
        needs_legacy_supplemental_data = any(
            params.get('enabled', False) and key in [
                'advanced_fund_features', 'chip_cost_breakthrough', 
                'chip_pressure_release', 'winner_rate_reversal', 'capital_flow_divergence'
            ]
            for key, params in indicators_config.items() if isinstance(params, dict)
        )
        if needs_legacy_supplemental_data:
            async def _fetch_legacy_supplemental_tagged(stock_code, trade_time, limit):
                trade_time_dt = pd.to_datetime(trade_time, utc=True) if trade_time else None
                df = await self.strategies_dao.get_fund_flow_and_chips_data(stock_code, trade_time_dt, limit)
                return ('legacy_supplemental', df)
            tasks.append(_fetch_legacy_supplemental_tagged(stock_code, trade_time, base_needed_bars))
            # print("    - [任务规划] 已添加“旧筹码与资金流”获取任务。")

        # 任务准备: 新筹码(AdvancedChipMetrics)
        chip_params = self._find_params_recursively(config, 'chip_feature_params')
        needs_advanced_chip_data = chip_params.get('enabled', False) if chip_params else False
        if needs_advanced_chip_data:
            async def _fetch_advanced_chips_tagged(stock_code, trade_time, limit):
                trade_time_dt = pd.to_datetime(trade_time, utc=True) if trade_time else None
                df = await self.strategies_dao.get_advanced_chip_metrics_data(stock_code, trade_time_dt, limit)
                return ('advanced_chips', df)
            tasks.append(_fetch_advanced_chips_tagged(stock_code, trade_time, base_needed_bars))
            # print("    - [任务规划] 已添加“新筹码(AdvancedChipMetrics)”获取任务。")

        # 4. 准备所有“基础”OHLCV数据获取任务
        async def _fetch_and_tag_data(tf_to_fetch, bars_to_fetch, trade_time_str):
            df = await self._get_ohlcv_data(stock_code, tf_to_fetch, bars_to_fetch, trade_time_str)
            return (tf_to_fetch, df)

        for tf in base_tfs_to_fetch:
            bars_to_fetch = base_needed_bars
            if 'D' in base_tfs_to_fetch and resample_map:
                bars_to_fetch = max(bars_to_fetch, 1200) # 约5年数据，确保周线/月线指标计算准确
            tasks.append(_fetch_and_tag_data(tf, bars_to_fetch, trade_time))
            # print(f"    - [任务规划] 已添加“OHLCV({tf})”获取任务，请求 {bars_to_fetch} 条数据。")

        # 5. 并发执行所有数据获取任务
        # print("    - [数据获取] 开始并发执行所有数据获取任务...")
        all_data_results = await asyncio.gather(*tasks, return_exceptions=True)
        # print("    - [数据获取] 所有任务执行完毕。")

        # 6. 分类处理获取到的数据
        raw_dfs: Dict[str, pd.DataFrame] = {}
        df_legacy_supplemental: Optional[pd.DataFrame] = None
        df_advanced_chips: Optional[pd.DataFrame] = None
        
        for result in all_data_results:
            if isinstance(result, Exception):
                print(f"      -> 警告: 一个数据获取任务失败: {result}")
                continue
            if not (isinstance(result, tuple) and len(result) == 2): continue
            
            tag, data = result
            if tag == 'legacy_supplemental':
                if isinstance(data, pd.DataFrame): df_legacy_supplemental = data
            elif tag == 'advanced_chips':
                if isinstance(data, pd.DataFrame): df_advanced_chips = data
            else: # 处理 OHLCV 数据
                if isinstance(data, pd.DataFrame) and not data.empty:
                    raw_dfs[tag] = data

        if 'D' not in raw_dfs:
            print(f"    - 错误: 最核心的日线数据获取失败，处理终止。")
            return {}
        
        print(f"    - [数据流追踪] 步骤1: 原始日线数据已加载，行数: {len(raw_dfs['D'])}")

        # 7. 【核心新增】执行重采样，锻造周线和月线OHLCV及核心指标
        if resample_map:
            df_daily = raw_dfs['D']
            for target_tf, source_tf in resample_map.items():
                if source_tf == 'D' and not df_daily.empty:
                    print(f"    - [数据锻造] 开始从日线数据合成 '{target_tf}' 周期数据...")
                    ohlc_rule = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
                    resample_period = 'W-FRI' if target_tf == 'W' else 'ME' # 'ME' for MonthEnd
                    df_resampled = df_daily.resample(resample_period).agg(ohlc_rule)
                    df_resampled.dropna(how='all', inplace=True)
                    
                    if not df_resampled.empty:
                        if target_tf == 'W':
                            df_synthetic_indicators = self._calculate_synthetic_weekly_indicators(df_daily, df_resampled)
                            df_resampled = df_resampled.merge(df_synthetic_indicators, left_index=True, right_index=True, how='left')

                        raw_dfs[target_tf] = df_resampled
                        print(f"      -> 合成完成，生成 {len(df_resampled)} 条 '{target_tf}' 周期记录。")

        # 8. 标准化所有周期的索引，并准备并发计算指标
        processed_dfs: Dict[str, pd.DataFrame] = {}
        calc_tasks = []

        async def _calculate_for_tf(tf, df):
            # print(f"    - [指标计算] 开始为周期 '{tf}' 准备并计算指标...")
            # 标准化索引
            df = self._standardize_df_index_to_utc(df)
            
            # 【核心】只为日线数据融合补充数据
            if tf == 'D':
                if df_legacy_supplemental is not None and not df_legacy_supplemental.empty:
                    df_legacy_std = self._standardize_df_index_to_utc(df_legacy_supplemental)
                    df = pd.merge(df, df_legacy_std, left_index=True, right_index=True, how='left')
                    df[list(df_legacy_std.columns)] = df[list(df_legacy_std.columns)].ffill()
                
                if df_advanced_chips is not None and not df_advanced_chips.empty:
                    df_advanced_chips_std = self._standardize_df_index_to_utc(df_advanced_chips)
                    df = pd.merge(df, df_advanced_chips_std, left_index=True, right_index=True, how='left')
                    df[list(df_advanced_chips_std.columns)] = df[list(df_advanced_chips_std.columns)].ffill()
            
            # 调用指标计算引擎
            df_with_indicators = await self._calculate_indicators_for_timescale(df, indicators_config, tf)
            # print(f"      -> 周期 '{tf}' 指标计算完成。")
            return tf, df_with_indicators

        for tf, df in raw_dfs.items():
            if tf in required_tfs:
                calc_tasks.append(_calculate_for_tf(tf, df))

        # 9. 并发执行所有指标计算任务
        processed_results = await asyncio.gather(*calc_tasks, return_exceptions=True)

        for res in processed_results:
            if isinstance(res, Exception):
                print(f"    - 错误: 指标计算任务中出现异常: {res}")
                continue
            if res and isinstance(res, tuple) and len(res) == 2:
                tf, df_processed = res
                if df_processed is not None and not df_processed.empty:
                    processed_dfs[tf] = df_processed
                else:
                    print(f"    - 警告: 周期 '{tf}' 的指标计算结果为空DataFrame，已被丢弃。")

        print(f"--- [数据准备V8.1] 数据准备完成，最终字典包含的周期: {sorted(list(processed_dfs.keys()))} ---")
        return processed_dfs

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

    def _calculate_synthetic_weekly_indicators(self, df_daily: pd.DataFrame, df_weekly: pd.DataFrame) -> pd.DataFrame:
        """
        【V8.0 新增】高级指标合成室
        专门用于从日线数据中，为周线数据合成那些依赖日内过程的复杂指标。
        
        Args:
            df_daily (pd.DataFrame): 完整的日线数据源。
            df_weekly (pd.DataFrame): 已经通过resample聚合好的周线OHLCV数据。

        Returns:
            pd.DataFrame: 一个包含新合成的周线指标列的DataFrame，其索引与df_weekly对齐。
        """
        print("      -> [高级指标合成室] 正在合成周线CMF等复杂指标...")
        synthetic_indicators = pd.DataFrame(index=df_weekly.index)

        # --- 1. 合成周线CMF (Chaikin Money Flow) ---
        # 步骤1.1: 在日线上计算每日的“资金流成交量” (Money Flow Volume)
        mfm = ((df_daily['close'] - df_daily['low']) - (df_daily['high'] - df_daily['close'])) / (df_daily['high'] - df_daily['low'])
        mfm = mfm.fillna(0)
        daily_mfv = mfm * df_daily['volume']

        # 步骤1.2: 将“日资金流成交量”和“日成交量”按周进行求和
        weekly_mfv_sum = daily_mfv.resample('W-FRI').sum()
        weekly_volume_sum = df_daily['volume'].resample('W-FRI').sum()

        # 步骤1.3: 在周线级别上，计算最终的21周CMF
        cmf_period = 21
        cmf_numerator = weekly_mfv_sum.rolling(window=cmf_period).sum()
        cmf_denominator = weekly_volume_sum.rolling(window=cmf_period).sum()
        
        # 避免除以零
        synthetic_indicators['CMF_21'] = np.divide(cmf_numerator, cmf_denominator, out=np.full_like(cmf_numerator, np.nan), where=cmf_denominator!=0)
        
        # --- 2. 未来可在此处添加更多复杂周线指标的合成逻辑 (如KDJ, RSI等) ---
        # 例如，合成周线RSI也应先计算每日的涨跌额，再按周聚合，最后计算RSI，以获得更平滑的结果。
        
        print("      -> [高级指标合成室] 合成完成。")
        return synthetic_indicators

    # ▼▼▼ 新增一个专门生成游资信号的函数 ▼▼▼
    async def _prepare_hot_money_signals(self, stock_code: str, start_date: datetime.date, end_date: datetime.date, params: dict) -> pd.DataFrame:
        """
        根据游资明细数据，生成一系列与日线数据对齐的原子信号。
        这些信号将被合并到日线DataFrame中。

        Args:
            stock_code (str): 股票代码。
            start_date (date): 数据查询的开始日期。
            end_date (date): 数据查询的结束日期。
            params (dict): 从配置文件中读取的 hot_money_params。

        Returns:
            pd.DataFrame: 一个包含多个布尔信号列的DataFrame，索引为日期。
        """
        print("    - [游资信号引擎] 开始准备游资原子信号...")
        
        # 从DAO获取原始游资数据
        hm_df = await self.fund_flow_dao.get_hm_detail_data(start_date, end_date, stock_codes=[stock_code])

        if hm_df.empty:
            print("    - [游资信号引擎] 无游资数据，返回空DataFrame。")
            return pd.DataFrame()

        hm_df['trade_date'] = pd.to_datetime(hm_df['trade_date'], utc=True)
        
        # 按交易日期聚合，计算每日的游资行为
        daily_summary = {}

        # --- 信号1: 当日有任意游资净买入 (HM_ACTIVE_ANY_D) ---
        any_buy_dates = hm_df[hm_df['net_amount'] > 0]['trade_date'].unique()
        daily_summary['HM_ACTIVE_ANY_D'] = pd.Series(True, index=any_buy_dates)
        
        # --- 信号2: 当日有顶级游资净买入 (HM_ACTIVE_TOP_TIER_D) ---
        top_tier_list = params.get('top_tier_list', [])
        top_tier_df = hm_df[hm_df['hm_name'].isin(top_tier_list)]
        top_tier_buy_dates = top_tier_df[top_tier_df['net_amount'] > 0]['trade_date'].unique()
        daily_summary['HM_ACTIVE_TOP_TIER_D'] = pd.Series(True, index=top_tier_buy_dates)

        # --- 信号3: 游资协同攻击 (HM_COORDINATED_ATTACK_D) ---
        coordination_threshold = params.get('coordination_threshold', 3)
        # 按天分组，计算当天净买入的独立游资家数
        buyers_count_daily = hm_df[hm_df['net_amount'] > 0].groupby('trade_date')['hm_name'].nunique()
        coordinated_dates = buyers_count_daily[buyers_count_daily >= coordination_threshold].index
        daily_summary['HM_COORDINATED_ATTACK_D'] = pd.Series(True, index=coordinated_dates)

        # 将所有信号合并成一个DataFrame
        signals_df = pd.DataFrame(daily_summary)
        
        print(f"      -> '任意游资活跃'信号: {daily_summary['HM_ACTIVE_ANY_D'].sum()} 天")
        print(f"      -> '顶级游资活跃'信号: {daily_summary['HM_ACTIVE_TOP_TIER_D'].sum()} 天")
        print(f"      -> '游资协同攻击'信号: {daily_summary['HM_COORDINATED_ATTACK_D'].sum()} 天")
        
        return signals_df

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
        【V110 终极规范版】根据配置为指定时间周期计算所有技术指标，并为所有列统一添加后缀。
        - 核心升级: 为所有时间周期（包括分钟线）统一添加后缀，如 '_5', '_30', '_D'。
        - 核心修复: 废弃了之前版本中对VWAP等指标的特殊处理，将其纳入统一的后缀添加流程，确保所有指标命名规则一致。
        """
        # print(f"  [指标计算V110] 开始为周期 '{timeframe_key}' 计算指标...")
        if not config:
            print(f"    - 警告: 周期 '{timeframe_key}' 没有配置任何指标。")
            return df

        max_required_period = self._get_max_period_for_timeframe(config, timeframe_key)
        if len(df) < max_required_period:
            logger.warning(f"数据行数 ({len(df)}) 不足以满足周期 '{timeframe_key}' 的最大计算要求 ({max_required_period})，将跳过该周期的所有指标计算。")
            return df

        # 创建一个副本用于计算，避免修改原始传入的DataFrame
        df_for_calc = df.copy()
        
        # 定义指标计算方法的映射
        indicator_method_map = {
            'ema': self.calculate_ema, 'vol_ma': self.calculate_vol_ma, 'trix': self.calculate_trix,
            'coppock': self.calculate_coppock, 'rsi': self.calculate_rsi, 'macd': self.calculate_macd,
            'dmi': self.calculate_dmi, 'roc': self.calculate_roc, 'boll_bands_and_width': self.calculate_boll_bands_and_width,
            'cmf': self.calculate_cmf, 'bias': self.calculate_bias, 'atrn': self.calculate_atrn,
            'atrr': self.calculate_atrr, 'obv': self.calculate_obv, 'kdj': self.calculate_kdj,
            'uo': self.calculate_uo, 'vwap': self.calculate_vwap, 'atr': self.calculate_atr,
            'consolidation_period': self.calculate_consolidation_period,
            'advanced_fund_features': self.calculate_advanced_fund_features,
            'fibonacci_levels': self.calculate_fibonacci_levels,
        }
        
        def merge_results(result_data, target_df):
            """健壮地合并指标计算结果（Series或DataFrame）到目标DataFrame。"""
            if result_data is None or result_data.empty: return
            if isinstance(result_data, pd.Series):
                result_data = result_data.to_frame()
            
            if isinstance(result_data, pd.DataFrame):
                for col in result_data.columns:
                    target_df[col] = result_data[col]
            else:
                logger.warning(f"指标计算返回了未知类型 {type(result_data)}，已跳过。")

        # --- 阶段一: 常规指标计算循环 ---
        for indicator_key, params in config.items():
            indicator_name = indicator_key.lower()
            
            # 跳过非指标项或禁用的指标
            if indicator_name in ['说明', 'index_sync', 'cyq_perf', 'zscore'] or not params.get('enabled', False): continue
            if indicator_name not in indicator_method_map:
                logger.warning(f"    - 警告: 未找到指标 '{indicator_name}' 的计算方法，已跳过。")
                continue
            # 复合指标在下一阶段处理
            if indicator_name in ['consolidation_period', 'advanced_fund_features', 'fibonacci_levels']: continue

            # 遍历该指标的所有配置项（例如，RSI可以有多个不同周期的配置）
            configs_to_process = params.get('configs', [params])
            for sub_config in configs_to_process:
                # 检查当前配置是否适用于正在处理的时间周期
                if timeframe_key not in sub_config.get("apply_on", []): continue
                
                try:
                    method_to_call = indicator_method_map[indicator_name]
                    kwargs = {'df': df_for_calc}
                    periods = sub_config.get('periods')

                    # VWAP的特殊处理：需要传递锚点
                    if indicator_name == 'vwap':
                        anchor = 'D' if timeframe_key.isdigit() else timeframe_key
                        kwargs['anchor'] = anchor
                        # 注意：此处不再传递suffix，因为所有后缀将在最后统一添加
                        result_df = await method_to_call(**kwargs)
                        merge_results(result_df, df_for_calc)
                        continue

                    if periods is None:
                        result_df = await method_to_call(**kwargs)
                        merge_results(result_df, df_for_calc)
                        continue
                    
                    # 处理多参数指标（如MACD, KDJ）和多周期配置
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
                            kwargs_iter['period'] = p_set[0] if isinstance(p_set, list) else p_set
                        
                        result_df = await method_to_call(**kwargs_iter)
                        merge_results(result_df, df_for_calc)
                except Exception as e:
                    logger.error(f"    - 计算指标 {indicator_name.upper()} (周期: {timeframe_key}, 参数: {sub_config.get('periods')}) 时出错: {e}", exc_info=True)

        # --- 阶段二: 复合指标计算循环 ---
        for indicator_key, params in config.items():
            indicator_name = indicator_key.lower()
            if indicator_name in ['consolidation_period', 'advanced_fund_features', 'fibonacci_levels'] and params.get('enabled', False):
                if timeframe_key in params.get("apply_on", []):
                    try:
                        method_to_call = indicator_method_map[indicator_name]
                        # 注意：此处传递空的suffix，因为后缀将统一添加
                        result_df = await method_to_call(df=df_for_calc, params=params, suffix='')
                        merge_results(result_df, df_for_calc)
                    except Exception as e:
                        logger.error(f"    - 复合指标 {indicator_name.upper()} (周期: {timeframe_key}) 计算时出错: {e}", exc_info=True)

        # --- 阶段三: 后处理指标计算（如Z-Score） ---
        zscore_params = config.get('zscore')
        if zscore_params and zscore_params.get('enabled', False):
            for z_config in zscore_params.get('configs', []):
                if timeframe_key not in z_config.get("apply_on", []): continue
                try:
                    source_pattern = z_config.get("source_column_pattern")
                    output_col_name = z_config.get("output_column_name")
                    window = z_config.get("window", 60)

                    # 动态构建源列名（不带后缀）
                    source_col_name = source_pattern
                    if "{fast}" in source_pattern:
                        macd_cfg = config.get('macd', {})
                        macd_periods = next((c.get('periods') for c in macd_cfg.get('configs', []) if timeframe_key in c.get('apply_on', [])), None)
                        if macd_periods:
                            source_col_name = source_pattern.format(fast=macd_periods[0], slow=macd_periods[1], signal=macd_periods[2])
                        else: continue
                    
                    # 移除日线后缀以匹配df_for_calc中的列名
                    source_col_name = source_col_name.removesuffix(f"_{timeframe_key}")
                    output_col_name = output_col_name.removesuffix(f"_{timeframe_key}")

                    if source_col_name in df_for_calc.columns:
                        source_series = df_for_calc[source_col_name]
                        rolling_mean = source_series.rolling(window=window).mean()
                        rolling_std = source_series.rolling(window=window).std()
                        zscore_result = np.divide((source_series - rolling_mean), rolling_std, out=np.full_like(source_series, np.nan), where=rolling_std!=0)
                        df_for_calc[output_col_name] = zscore_result
                    else:
                        logger.warning(f"Z-score计算失败：源列 '{source_col_name}' 在临时DataFrame中不存在。")
                except Exception as e:
                    logger.error(f"计算Z-score时出错: {e}", exc_info=True)

        # --- 阶段四: 统一添加后缀并返回 ---
        # ▼▼▼【代码修改 V110】: 统一为所有列添加后缀的核心逻辑 ▼▼▼
        # 1. 定义后缀
        suffix = f"_{timeframe_key}"
        
        # 2. 为 df_for_calc 中的每一列（包括原始OHLCV和所有计算出的指标）都规划好带后缀的新名字。
        rename_map = {col: f"{col}{suffix}" for col in df_for_calc.columns}
        
        # 3. 应用重命名，生成最终的DataFrame
        final_df = df_for_calc.rename(columns=rename_map)

        # 4. 调试打印
        # print(f"\n--- [IndicatorService V110 调试输出] 周期 '{timeframe_key}' 最终生成列名清单 (已添加后缀) ---")
        # print(final_df.columns.tolist())
        # print(f"--- [IndicatorService V110 调试输出结束] ---\n")

        return final_df

    async def _calculate_all_slopes(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V2.0 跨周期生产线版】
        - 职责: 作为数据工程的一部分，为所有指定的时间周期计算斜率和加速度。
        """
        print("    - [斜率中心 V2.0 跨周期生产线版 @ IndicatorService] 启动...")
        # 注意：这里的参数路径可能需要根据IndicatorService的上下文调整
        # 假设config就是完整的策略配置
        slope_params = config.get('feature_engineering_params', {}).get('slope_params', {})
        if not slope_params.get('enabled', False):
            print("      -> 斜率计算被禁用，跳过。")
            return all_dfs

        series_to_slope = slope_params.get('series_to_slope', {})
        if not series_to_slope:
            print("      -> [信息] 未在配置中指定任何需要计算斜率的序列，跳过。")
            return all_dfs

        # (这里是之前改造好的、完整的跨周期斜率计算逻辑)
        for col_pattern, lookbacks in series_to_slope.items():
            try:
                timeframe = col_pattern.split('_')[-1]
                if timeframe.upper() not in ['D', 'W', 'M', 'Q', 'Y'] and not timeframe.isdigit():
                    timeframe = 'D'
            except IndexError:
                continue

            if timeframe not in all_dfs:
                continue
            
            df = all_dfs[timeframe]
            
            if col_pattern not in df.columns:
                continue

            # print(f"      -> 正在为周期 '{timeframe}' 的指标 '{col_pattern}' 计算斜率...")
            source_series = df[col_pattern].astype(float)
            
            newly_created_slope_cols = []

            for lookback in lookbacks:
                slope_col_name = f'SLOPE_{lookback}_{col_pattern}'
                if slope_col_name in df.columns: continue
                min_p = max(2, lookback // 2)
                linreg_result = df.ta.linreg(close=source_series, length=lookback, min_periods=min_p, slope=True, intercept=False, r=False)
                slope_series = linreg_result if isinstance(linreg_result, pd.Series) else linreg_result.iloc[:, 0]
                df[slope_col_name] = slope_series.fillna(0)
                newly_created_slope_cols.append((slope_col_name, lookback, col_pattern))

            for slope_col_name, lookback, original_col_name in newly_created_slope_cols:
                accel_col_name = f'ACCEL_{lookback}_{original_col_name}'
                if accel_col_name in df.columns: continue
                if not df[slope_col_name].dropna().empty:
                    min_p = max(2, lookback // 2)
                    accel_linreg_result = df.ta.linreg(close=df[slope_col_name], length=lookback, min_periods=min_p, slope=True, intercept=False, r=False)
                    accel_series = accel_linreg_result if isinstance(accel_linreg_result, pd.Series) else accel_linreg_result.iloc[:, 0]
                    df[accel_col_name] = accel_series.fillna(0)
                else:
                    df[accel_col_name] = np.nan
            
            all_dfs[timeframe] = df

        print("    - [斜率中心 V2.0 @ IndicatorService] 所有斜率相关计算完成。")
        return all_dfs

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

    async def calculate_boll_bands_and_width(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, close_col='close', suffix: str = '') -> Optional[pd.DataFrame]:
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

            bbands_df = await asyncio.to_thread(lambda: ta.bbands(close=df[close_col], length=period, std=std_dev, append=False))
            if bbands_df is None or bbands_df.empty: return None

            bbw_source_col = f'BBB_{period}_{std_dev:.1f}'
            if bbw_source_col in bbands_df.columns:
                bbands_df[bbw_source_col] = bbands_df[bbw_source_col] / 100.0
            rename_map = {
                f'BBL_{period}_{std_dev:.1f}': f'BBL_{period}_{std_dev:.1f}{suffix}',
                f'BBM_{period}_{std_dev:.1f}': f'BBM_{period}_{std_dev:.1f}{suffix}',
                f'BBU_{period}_{std_dev:.1f}': f'BBU_{period}_{std_dev:.1f}{suffix}',
                bbw_source_col: f'BBW_{period}_{std_dev:.1f}{suffix}',
                f'BBP_{period}_{std_dev:.1f}': f'BBP_{period}_{std_dev:.1f}{suffix}'
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
        min_length = period + signal_period + smooth_k_period
        if len(df) < min_length:
            print(f"调试信息: 数据长度 {len(df)} 小于计算KDJ所需的最小长度 {min_length}，跳过计算。")
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

    async def calculate_vwap(self, df: pd.DataFrame, anchor: Optional[str] = None, suffix: str = '') -> Optional[pd.DataFrame]:
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
            def _sync_vwap():
                return df.ta.vwap(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], anchor=processed_anchor, append=False)
            
            vwap_series = await asyncio.to_thread(_sync_vwap)
            if vwap_series is None or vwap_series.empty: return None

            # pandas-ta的vwap列名比较特殊，我们手动重命名以确保一致性
            # 原始列名可能是 VWAP_D, VWAP_W, VWAP_30T 等
            original_name = vwap_series.name
            # 我们统一将其命名为 VWAP_{suffix}
            new_name = f'VWAP{suffix}'
            vwap_series.name = new_name
            
            return pd.DataFrame(vwap_series)
        except Exception as e:
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
