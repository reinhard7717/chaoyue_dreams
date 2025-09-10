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
from .indicator_calculate_services import IndicatorCalculator
from dao_manager.tushare_daos.industry_dao import IndustryDao
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.index_basic_dao import IndexBasicDAO
from core.constants import TimeLevel
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO
from dao_manager.tushare_daos.fund_flow_dao import FundFlowDao
from utils.cache_manager import CacheManager
from utils.math_tools import hurst_exponent

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
    def __init__(self, cache_manager_instance: CacheManager):
        """
        【V8.1 依赖注入版】初始化 IndicatorService。
        - 接收 CacheManager 实例，并将其注入所有内部创建的DAO。
        """
        # 将 cache_manager_instance 传递给所有DAO的构造函数
        self.indicator_dao = IndicatorDAO(cache_manager_instance)
        self.industry_dao = IndustryDao(cache_manager_instance)
        self.stock_basic_dao = StockBasicInfoDao(cache_manager_instance)
        self.stock_trade_dao = StockTimeTradeDAO(cache_manager_instance)
        self.index_dao = IndexBasicDAO(cache_manager_instance)
        self.strategies_dao = StrategiesDAO(cache_manager_instance)
        self.fund_flow_dao = FundFlowDao(cache_manager_instance)
        self.calculator = IndicatorCalculator()

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
        # 预处理 time_level 字符串，移除 'min' 后缀，以匹配 DAO 层的模型查找逻辑
        processed_time_level = str(time_level).lower()
        if processed_time_level.endswith('min'):
            processed_time_level = processed_time_level.replace('min', '')
        # ▼▼▼【代码修改 V117.28】: 确保获取实时数据 ▼▼▼
        # 如果 trade_time 未提供（例如在实时触发的场景），则使用当前时间作为查询终点。
        # 这确保了DAO层能够获取到截至目前的最新数据，包括当天的盘中K线。
        df = await self.indicator_dao.get_history_ohlcv_df(
            stock_code=stock_code, 
            time_level=processed_time_level,
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

    def _rename_precomputed_derivatives(self, df: pd.DataFrame) -> pd.DataFrame: # 整个方法
        """
        【新增 V1.0】预计算衍生指标列名适配器
        - 核心职责: 将从数据库加载的、已持久化的衍生指标列名（如 'peak_cost_slope_5d'）
                    转换为策略层期望的、历史悠久的命名格式（如 'SLOPE_5_peak_cost_D'）。
                    这是适配“衍生指标持久化”重构的关键一步，确保上层策略代码无需修改。
        - 输入: 包含数据库原始列名的DataFrame (来自 advanced_chips)。
        - 输出: 列名已转换为策略层格式的DataFrame。
        """
        # print("    - [数据适配层] 正在转换预计算的衍生指标列名...")
        import re # 导入正则表达式模块
        rename_map = {}
        for col in df.columns:
            # 匹配斜率列，例如: peak_cost_slope_5d
            slope_match = re.match(r'(.+)_slope_(\d+)d$', col)
            if slope_match:
                base_name = slope_match.group(1)
                period = slope_match.group(2)
                # 转换为: SLOPE_5_peak_cost_D
                new_name = f"SLOPE_{period}_{base_name}_D"
                rename_map[col] = new_name
                continue # 匹配成功后继续下一个循环

            # 匹配加速度列，例如: peak_control_ratio_accel_21d
            accel_match = re.match(r'(.+)_accel_(\d+)d$', col)
            if accel_match:
                base_name = accel_match.group(1)
                period = accel_match.group(2)
                # 转换为: ACCEL_21_peak_control_ratio_D
                new_name = f"ACCEL_{period}_{base_name}_D"
                rename_map[col] = new_name
        
        if rename_map:
            # print(f"      -> 发现并转换 {len(rename_map)} 个衍生指标列。")
            return df.rename(columns=rename_map)
        else:
            print("      -> 未发现需要转换的衍生指标列。")
            return df

    def _get_max_lookback_period(self, config: dict) -> int:
        """
        【军需官】扫描整个策略配置，找出所有指标中要求的最长回溯期。
        这是一个简化的实现，用于演示核心思想。
        """
        # print("    - [军需官] 正在扫描全军军火库，确定最大回溯需求...")
        # 简化实现：
        calculated_max = 350 # 保守估计，足以满足EMA(55周)等大周期指标
        print(f"    - [军需官] 扫描完成，最大回溯需求估算为 {calculated_max} 个日线周期。")
        return calculated_max

    async def prepare_data_for_strategy(
        self,
        stock_code: str,
        config: dict,
        trade_time: Optional[str] = None,
        latest_only: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        【V8.0 情报锻造中心版】为策略准备数据的统一入口。
        此版本将调用重构后的核心数据准备函数。
        """
        # print(f"--- [数据准备V8.0-情报锻造中心] 开始为 {stock_code} 准备数据... ---")

        # --- 步骤 1: 【第一道工序】准备基础数据和常规指标 ---
        all_dfs = await self._prepare_base_data_and_indicators(stock_code, config, trade_time, latest_only=latest_only)
        if not all_dfs:
            return {}
        # --- 步骤 2: 【第二道工序】计算元特征 (Hurst, CV等) ---
        all_dfs = await self._calculate_meta_features(all_dfs, config)
        # --- 步骤 3: 【VPA效率指标计算】 - 修正顺序，提前计算 ---
        # 解释：高级模式识别依赖VPA效率指标，因此必须先计算VPA。
        all_dfs = await self._calculate_vpa_features(all_dfs, config)
        # --- 步骤 4: 【高级模式识别】 - 移至VPA计算之后 ---
        all_dfs = await self._calculate_pattern_recognition_signals(all_dfs, config)
        # --- 步骤 5: 【斜率计算】 - 修正步骤编号 ---
        # 此刻，hurst_60d_D, price_cv_60d_D 等列已经存在，可以安全地计算它们的斜率了
        all_dfs = await self._calculate_all_slopes(all_dfs, config)
        # --- 步骤 6: 【加速度计算】 - 修正步骤编号 ---
        all_dfs = await self._calculate_all_accelerations(all_dfs, config)
        # --- 步骤 7: 【上下文信息注入】 - 修正步骤编号 ---
        if not all_dfs or 'D' not in all_dfs or all_dfs['D'].empty:
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
        
        #  调用军械库清单生成器 ▼▼▼
        # self._log_final_data_columns(all_dfs)
        
        return all_dfs

    async def _prepare_base_data_and_indicators(
        self,
        stock_code: str,
        config: dict,
        trade_time: Optional[str] = None,
        latest_only: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        【V8.8 双向对齐终极版】
        - 核心修正: 在合并前，强制将主数据和所有补充数据的DatetimeIndex都标准化到午夜(normalize)。
                    这确保了无论原始数据的时间戳是收盘时刻还是午夜，都能实现精确的日期对齐，
                    从根本上杜绝了因时间部分不一致导致的合并失败和数据错位风险。
        - 解决方案: 仅对多来源且存在同名列的资金流数据（fund_flow_*）添加来源后缀，确保所有资金流数据被完整保留。
                   对于其他补充数据（如daily_basic），则沿用“移除冲突列”的策略，以保护主OHLCV数据的权威性。
        - 优化: 对整个方法进行了代码审查，并添加了详尽的中文注释，解释了每一步的逻辑。
        """
        # 更新版本号和日志信息
        print(f"--- [数据准备V8.8] 开始为 {stock_code} 准备基础数据与指标 ---")
        # --- 步骤 1: 解析配置，确定需要计算的时间周期 ---
        # 从策略配置文件中，找出所有需要生成指标的时间周期（如 'D', 'W', 'M'）
        required_tfs = self._discover_required_timeframes_from_config(config)
        if not required_tfs:
            # 如果配置中没有任何时间周期要求，则直接返回空字典，终止处理
            print("    - [配置读取] 未发现任何需要的时间周期，处理终止。")
            return {}
        # --- 步骤 2: 确定数据加载量 ---
        # 判断是否为“闪电模式”（latest_only=True），该模式仅用于最新交易日的快速信号生成
        if latest_only:
            # 在闪电模式下，计算策略所需的最大回溯期，并增加一个安全缓冲期
            max_lookback = self._get_max_lookback_period(config)
            safety_buffer = 100 
            base_needed_bars = max_lookback + safety_buffer
            print(f"    - [闪电模式启动] 策略最大回溯期: {max_lookback}, 安全缓冲: {safety_buffer}, 最终加载: {base_needed_bars} 条记录。")
        else:
            # 在完整回测或常规模式下，加载配置文件中指定的默认数据量（通常较大，如1200条）
            base_needed_bars = config.get('feature_engineering_params', {}).get('base_needed_bars', 1200)
        # --- 步骤 3: 规划数据获取策略 ---
        # base_tfs_to_fetch: 存储需要直接从API获取的基础时间周期数据
        # resample_map: 存储需要通过重采样生成的目标时间周期及其来源（如 {'W': 'D'}）
        base_tfs_to_fetch = set()
        resample_map = {} 
        for tf in required_tfs:
            if tf in ['W', 'M']:  # 周线和月线数据由日线数据重采样生成
                base_tfs_to_fetch.add('D') 
                resample_map[tf] = 'D'
            else:  # 其他时间周期（如 'D', '60T'）直接获取
                base_tfs_to_fetch.add(tf)
        # --- 步骤 4: 准备所有补充数据的异步获取任务 ---
        # 根据策略配置，判断需要哪些类型的补充数据（旧版筹码、高级筹码、日度基本面、各类资金流等）
        indicators_config = config.get('feature_engineering_params', {}).get('indicators', {})
        tasks = [] # 用于收集所有异步任务的列表
        # 检查是否需要旧版的资金流和筹码数据
        needs_legacy_supplemental_data = any(
            params.get('enabled', False) and key in [
                'chip_cost_breakthrough', 'chip_pressure_release', 'winner_rate_reversal', 'capital_flow_divergence'
            ]
            for key, params in indicators_config.items() if isinstance(params, dict)
        )
        if needs_legacy_supplemental_data:
            async def _fetch_legacy_supplemental_tagged(stock_code, trade_time, limit):
                trade_time_dt = pd.to_datetime(trade_time, utc=True) if trade_time else None
                df = await self.strategies_dao.get_fund_flow_and_chips_data(stock_code, trade_time_dt, limit)
                return ('legacy_supplemental', df) # 返回一个元组，包含数据标识和DataFrame
            tasks.append(_fetch_legacy_supplemental_tagged(stock_code, trade_time, base_needed_bars))
        # 检查是否需要高级筹码指标数据
        chip_params = self._find_params_recursively(config, 'chip_feature_params')
        needs_advanced_chip_data = chip_params.get('enabled', False) if chip_params else False
        if needs_advanced_chip_data:
            async def _fetch_advanced_chips_tagged(stock_code, trade_time, limit):
                trade_time_dt = pd.to_datetime(trade_time, utc=True) if trade_time else None
                df = await self.strategies_dao.get_advanced_chip_metrics_data(stock_code, trade_time_dt, limit)
                return ('advanced_chips', df)
            tasks.append(_fetch_advanced_chips_tagged(stock_code, trade_time, base_needed_bars))
        # 日度基本面数据是常用数据，默认获取
        async def _fetch_daily_basic_tagged(stock_code, trade_time, limit):
            trade_time_dt = pd.to_datetime(trade_time, utc=True) if trade_time else None
            df = await self.strategies_dao.get_daily_basic_data(stock_code, trade_time_dt, limit)
            return ('daily_basic', df)
        tasks.append(_fetch_daily_basic_tagged(stock_code, trade_time, base_needed_bars))
        trade_time_dt_date = pd.to_datetime(trade_time, utc=True).date() if trade_time else datetime.datetime.now().date()
        # 同花顺资金流
        async def _fetch_fund_flow_ths_tagged(stock_code, trade_time_dt_date, limit):
            df = await self.fund_flow_dao.get_fund_flow_ths_data(stock_code, trade_time_dt_date, limit)
            return ('fund_flow_ths', df)
        tasks.append(_fetch_fund_flow_ths_tagged(stock_code, trade_time_dt_date, base_needed_bars))
        # 东方财富资金流
        async def _fetch_fund_flow_dc_tagged(stock_code, trade_time_dt_date, limit):
            df = await self.fund_flow_dao.get_fund_flow_dc_data(stock_code, trade_time_dt_date, limit)
            return ('fund_flow_dc', df)
        tasks.append(_fetch_fund_flow_dc_tagged(stock_code, trade_time_dt_date, base_needed_bars))
        # Tushare资金流
        async def _fetch_fund_flow_tushare_tagged(stock_code, trade_time_dt_date, limit):
            df = await self.fund_flow_dao.get_fund_flow_daily_data(stock_code, trade_time_dt_date, limit)
            return ('fund_flow_tushare', df)
        tasks.append(_fetch_fund_flow_tushare_tagged(stock_code, trade_time_dt_date, base_needed_bars))
        # 增加获取高级资金指标的任务
        async def _fetch_advanced_fund_flow_tagged(stock_code, trade_time_dt_date, limit):
            df = await self.fund_flow_dao.get_advanced_fund_flow_metrics_data(stock_code, trade_time_dt_date, limit)
            return ('advanced_fund_flow', df)
        tasks.append(_fetch_advanced_fund_flow_tagged(stock_code, trade_time_dt_date, base_needed_bars))
        # --- 步骤 5: 准备基础OHLCV数据的异步获取任务 ---
        async def _fetch_and_tag_data(tf_to_fetch, trade_time_str):
            df = await self._get_ohlcv_data(stock_code, tf_to_fetch, base_needed_bars, trade_time_str)
            return (tf_to_fetch, df)
        for tf in base_tfs_to_fetch:
            tasks.append(_fetch_and_tag_data(tf, trade_time))
        # --- 步骤 6: 并发执行所有数据获取任务 ---
        # 使用 asyncio.gather 并发运行所有收集到的任务，提高数据获取效率
        all_data_results = await asyncio.gather(*tasks, return_exceptions=True)
        # --- 步骤 7: 分类和预处理获取到的数据 ---
        raw_dfs: Dict[str, pd.DataFrame] = {} # 存放基础OHLCV数据
        supplemental_dfs: Dict[str, pd.DataFrame] = {} # 存放所有补充数据
        for result in all_data_results:
            if isinstance(result, Exception):
                print(f"      -> 警告: 一个数据获取任务失败: {result}")
                continue
            if not (isinstance(result, tuple) and len(result) == 2): continue
            tag, data = result # 解包任务返回的标识和DataFrame
            if isinstance(data, pd.DataFrame) and not data.empty:
                # 统一数据类型，防止Decimal与float运算冲突
                object_cols = data.select_dtypes(include=['object']).columns
                for col in object_cols:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                # 根据tag将数据分类存入不同的字典
                if tag in ['legacy_supplemental', 'advanced_chips', 'daily_basic', 'fund_flow_ths', 'fund_flow_dc', 'fund_flow_tushare', 'advanced_fund_flow']:
                    supplemental_dfs[tag] = data
                else:
                    raw_dfs[tag] = data
        # 核心的日线数据是所有计算的基础，如果获取失败则无法继续
        if 'D' not in raw_dfs:
            print(f"    - 错误: 最核心的日线数据获取失败，处理终止。")
            return {}
        # print(f"    - [数据流追踪] 步骤1: 原始日线数据已加载，行数: {len(raw_dfs['D'])}")
        # --- 步骤 8: 【核心逻辑】合并所有日级别数据 ---
        # 将所有补充数据合并到主日线DataFrame中，形成一个包含所有信息的“大师版”日线数据
        df_daily_master = raw_dfs['D']
        # 日期对齐修复：在合并前，将主数据的索引标准化到午夜，消除时间部分差异。
        df_daily_master.index = df_daily_master.index.normalize()
        for tag, df_supp in supplemental_dfs.items():
            # 标准化补充数据的索引为UTC时区，以便与主数据对齐
            df_supp_std = self._standardize_df_index_to_utc(df_supp)
            if df_supp_std is not None and not df_supp_std.empty:
                # 日期对齐修复：同样将补充数据的索引标准化到午夜，确保双向对齐。
                df_supp_std.index = df_supp_std.index.normalize()
                # 当处理高级筹码数据时，调用列名适配器
                if tag == 'advanced_chips' or tag == 'advanced_fund_flow':
                    df_supp_std = self._rename_precomputed_derivatives(df_supp_std)

                # 仅对 fund_flow_dao 相关的数据源添加后缀，因为它们之间存在大量同名列，需要区分来源
                if tag in ['fund_flow_ths', 'fund_flow_dc', 'fund_flow_tushare']:
                    suffix = f"_{tag}"
                    df_supp_std = df_supp_std.add_suffix(suffix)
                    # print(f"    - [数据合并] 为资金流数据源 '{tag}' 的所有列添加后缀 '{suffix}'。")
                else:
                    # 对于其他补充数据（如daily_basic），采用移除冲突列的保守策略，以保证OHLCV数据的权威性
                    conflicting_cols = df_daily_master.columns.intersection(df_supp_std.columns)
                    if not conflicting_cols.empty:
                        # print(f"    - [数据合并] 在 '{tag}' 数据中发现冲突列: {list(conflicting_cols)}，将从补充数据中移除。")
                        df_supp_std = df_supp_std.drop(columns=conflicting_cols)
                # 获取处理后真正要被合并的新列名
                new_cols_to_merge = df_supp_std.columns
                if new_cols_to_merge.empty:
                    # print(f"    - [数据合并] '{tag}' 数据在处理后无新列可合并，跳过。")
                    continue
                # 使用 'left' join 将处理后的补充数据合并到主日线数据中
                df_daily_master = pd.merge(df_daily_master, df_supp_std, left_index=True, right_index=True, how='left')
                # 对新合并的列进行前向填充（ffill），处理因节假日等原因造成的缺失值
                df_daily_master[list(new_cols_to_merge)] = df_daily_master[list(new_cols_to_merge)].ffill()
        # 用合并后的“大师版”日线数据替换原始的纯OHLCV日线数据
        raw_dfs['D'] = df_daily_master
        # print(f"    - [数据流追踪] 步骤2: 所有日级别数据已合并，主日线现有列数: {len(df_daily_master.columns)}")
        # --- 步骤 9: 执行重采样，生成周线和月线数据 ---
        if resample_map:
            df_daily = raw_dfs['D'] # 现在 df_daily 是包含所有信息的“大师版”
            for target_tf, source_tf in resample_map.items():
                if source_tf == 'D' and not df_daily.empty:
                    # 创建一个全面的聚合规则字典
                    aggregation_rules = {
                        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                    }
                    # 动态地为所有金额、交易量、净流入相关的列设置 'sum' 规则
                    for col in df_daily.columns:
                        if 'amount' in col.lower() or 'vol' in col.lower() or 'net' in col.lower():
                            if col not in aggregation_rules:
                                aggregation_rules[col] = 'sum'
                    # 动态地为所有比率相关的列设置 'last' 规则
                    for col in df_daily.columns:
                        if 'rate' in col.lower():
                            if col not in aggregation_rules:
                                aggregation_rules[col] = 'last'
                    # 为所有高级筹码指标（通常以 _D 结尾且不是OHLCV）自动添加 'last' 聚合规则
                    # 这样可以确保它们在生成周线数据时被保留下来，其值为每周最后一天的值
                    chip_related_keywords = ['chip_', 'concentration', 'peak_', 'winner', 'pressure', 'support', 'turnover_from']
                    fund_flow_keywords = ['fund_flow', 'consensus', 'divergence', 'main_force', 'retail']
                    for col in df_daily.columns:
                        if any(keyword in col.lower() for keyword in chip_related_keywords + fund_flow_keywords):
                            if col not in aggregation_rules:
                                aggregation_rules[col] = 'last'
                    if 'turnover_rate' in aggregation_rules:
                        aggregation_rules['turnover_rate'] = 'mean'
                    resample_period = 'W-FRI' if target_tf == 'W' else 'ME'
                    df_resampled = df_daily.resample(resample_period).agg(aggregation_rules)
                    df_resampled.dropna(how='all', inplace=True)
                    if not df_resampled.empty:
                        if target_tf == 'W':
                            df_synthetic_indicators = self._calculate_synthetic_weekly_indicators(df_daily, df_resampled)
                            df_resampled = df_resampled.merge(df_synthetic_indicators, left_index=True, right_index=True, how='left')
                        raw_dfs[target_tf] = df_resampled
                        print(f"      -> 合成完成，生成 {len(df_resampled)} 条 '{target_tf}' 周期记录，列数: {len(df_resampled.columns)}")
        # --- 步骤 10: 并发计算所有时间周期的技术指标 ---
        processed_dfs: Dict[str, pd.DataFrame] = {}
        calc_tasks = []
        async def _calculate_for_tf(tf, df):
            df = self._standardize_df_index_to_utc(df)
            df_with_indicators = await self._calculate_indicators_for_timescale(df, indicators_config, tf)
            return tf, df_with_indicators
        for tf, df in raw_dfs.items():
            if tf in required_tfs:
                calc_tasks.append(_calculate_for_tf(tf, df))
        # --- 步骤 11: 收集并整理指标计算结果 ---
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
        # 返回最终处理好的数据字典
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
        # print("      -> [高级指标合成室] 正在合成周线CMF等复杂指标...")
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
        synthetic_indicators['CMF_21_W'] = np.divide(cmf_numerator, cmf_denominator, out=np.full_like(cmf_numerator, np.nan), where=cmf_denominator!=0)
        
        # --- 2. 合成周线RSI (Relative Strength Index) ---
        # 步骤2.1: 在日线上计算每日的价格变动
        delta = df_daily['close'].diff()
        # 步骤2.2: 分离每日的上涨(gain)和下跌(loss)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        # 步骤2.3: 将每日的上涨和下跌按周进行求和
        weekly_gain_sum = gain.resample('W-FRI').sum()
        weekly_loss_sum = loss.resample('W-FRI').sum()
        # 步骤2.4: 在周线级别上，计算最终的13周RSI
        rsi_period = 13
        # 使用 pandas_ta 的 rma (Wilder's Moving Average) 来计算平均增益和平均损失，这是计算RSI的标准方法
        avg_gain = ta.rma(weekly_gain_sum, length=rsi_period)
        avg_loss = ta.rma(weekly_loss_sum, length=rsi_period)
        # 步骤2.5: 计算相对强度(RS)
        rs = avg_gain / (avg_loss + 1e-9) # 加上一个极小值防止除以零
        # 步骤2.6: 计算RSI
        rsi = 100 - (100 / (1 + rs))
        # 将结果添加到DataFrame中，注意不要加后缀
        synthetic_indicators['RSI_13_W'] = rsi
        # print("      -> [高级指标合成室] 合成完成。")
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

    async def _calculate_pattern_recognition_signals(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V2.0 多因子共振版】高级模式识别信号生产线
        - 核心职责: 基于波动率、趋势、资金流、筹码结构等多维度指标的共振，精确识别市场关键阶段。
        - 优化逻辑:
          - is_consolidation_D: 综合 BBW、ATR、ADX、均线粘合度进行判断。
          - is_breakthrough_D: 要求在盘整后发生，并有VPA效率和主力资金共识的确认。
          - is_accumulation_D: 核心是识别“价平量增”和“主力买、散户卖”的资金流背离特征。
          - is_distribution_D: 识别“高位滞涨派发”和“盘整期阴跌派发”两种典型场景。
        """
        # print("    - [高级模式识别生产线 V2.0] 启动...")
        timeframe = 'D'
        if timeframe not in all_dfs:
            return all_dfs
        df = all_dfs[timeframe]
        # --- 1. 军备检查 (升级版) ---
        # 检查计算所需的依赖列是否都存在
        required_cols = [
            'high_D', 'low_D', 'close_D', 'volume_D', 'pct_change_D',
            'BBW_21_2.0_D', 'ATR_14_D', 'MA_CONV_CV_SHORT_D', 'CMF_21_D',
            'VPA_EFFICIENCY_D', 'main_force_net_flow_consensus_D',
            'flow_divergence_mf_vs_retail_D', 'concentration_90pct_D',
            'winner_profit_margin_D', 'dynamic_consolidation_high_D', 'dynamic_consolidation_low_D'
        ]
        # 注意: ADX 来自 DMI 计算，这里假设它已存在。如果不存在，需要确保 calculate_dmi 被调用。
        # 为了健壮性，我们动态检查 ADX
        adx_col = next((col for col in df.columns if col.startswith('ADX_')), None)
        if adx_col:
            required_cols.append(adx_col)
        else:
            print("      -> [警告] 未找到 ADX 列，盘整识别的准确性会受影响。")
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"      -> [严重警告] 高级模式识别生产线缺少关键数据: {missing}，模块已跳过！")
            return all_dfs
        # --- 2. 计算 is_consolidation_D (盘整期) ---
        # 条件1: 波动率收缩 (布林带宽度或ATR处于近期低位)
        bbw_quantile = df['BBW_21_2.0_D'].rolling(window=60, min_periods=20).quantile(0.20)
        atr_quantile = df['ATR_14_D'].rolling(window=60, min_periods=20).quantile(0.20)
        cond_low_volatility = (df['BBW_21_2.0_D'] < bbw_quantile) | (df['ATR_14_D'] < atr_quantile)
        # 条件2: 趋势不明朗 (ADX低于25 或 均线高度粘合)
        cond_no_trend = (df[adx_col] < 25) if adx_col else pd.Series(True, index=df.index) # 如果没有ADX，则放宽条件
        cond_ma_converged = df['MA_CONV_CV_SHORT_D'] < 0.01 # 均线离散度小于1%
        is_consolidation = cond_low_volatility & (cond_no_trend | cond_ma_converged)
        df['is_consolidation_D'] = is_consolidation
        # --- 3. 计算 is_breakthrough_D (向上突破) & is_breakdown_D (向下跌破) ---
        # 突破条件
        was_consolidating = df['is_consolidation_D'].shift(1).fillna(False)
        price_break_box = df['close_D'] > df['dynamic_consolidation_high_D'].shift(1)
        volume_confirms = df['volume_D'] > df['VOL_MA_21_D'] * 1.2 # 成交量放大20%
        vpa_confirms = df['VPA_EFFICIENCY_D'] > 0.5 # 资金攻击效率较高
        money_flow_confirms = (df['CMF_21_D'] > 0.05) & (df['main_force_net_flow_consensus_D'] > 0)
        is_breakthrough = was_consolidating & price_break_box & volume_confirms & vpa_confirms & money_flow_confirms
        df['is_breakthrough_D'] = is_breakthrough
        # 跌破条件
        price_breakdown_box = df['close_D'] < df['dynamic_consolidation_low_D'].shift(1)
        is_breakdown = was_consolidating & price_breakdown_box & volume_confirms
        df['is_breakdown_D'] = is_breakdown
        # --- 4. 计算 is_accumulation_D (吸筹期) & is_distribution_D (派发期) ---
        # 吸筹 = 盘整期 + 主力买散户卖 + 筹码集中度上升
        cond_accumulation_flow = (df['flow_divergence_mf_vs_retail_D'] > 0.1).rolling(window=3).sum() == 3
        concentration_slope = df['concentration_90pct_D'].diff()
        cond_concentration_increase = (concentration_slope > 0).rolling(window=5).sum() >= 3 # 近5天有3天以上筹码在集中
        df['is_accumulation_D'] = is_consolidation & (cond_accumulation_flow | cond_concentration_increase)
        # 派发场景1: 高位滞涨派发 (天量不涨或微涨)
        high_volume = df['volume_D'] > df['VOL_MA_21_D'] * 2.0 # 成交量超过2倍均量
        stagnant_price = df['pct_change_D'].abs() < 0.01 # 涨跌幅小于1%
        high_winner_margin = df['winner_profit_margin_D'] > 30 # 获利盘丰厚
        low_vpa_efficiency = df['VPA_EFFICIENCY_D'] < 0.1
        dist_at_top = high_volume & (stagnant_price | low_vpa_efficiency) & high_winner_margin
        # 派发场景2: 盘整期派发 (主力卖散户买)
        cond_distribution_flow = (df['flow_divergence_mf_vs_retail_D'] < -0.1).rolling(window=3).sum() == 3
        dist_in_consolidation = is_consolidation & cond_distribution_flow
        df['is_distribution_D'] = dist_at_top | dist_in_consolidation
        # --- 5. 确保所有列都为布尔型 ---
        pattern_cols = ['is_consolidation_D', 'is_breakthrough_D', 'is_breakdown_D', 'is_accumulation_D', 'is_distribution_D']
        for col in pattern_cols:
            if col in df.columns:
                df[col] = df[col].fillna(False).astype(bool)
                # print(f"      -> 信号 '{col}' 已生成，共激活 {df[col].sum()} 天。")
        all_dfs[timeframe] = df
        print("    - [高级模式识别生产线 V2.0] 所有模式信号生产完成。")
        return all_dfs

    async def _calculate_indicators_for_timescale(self, df: pd.DataFrame, config: dict, timeframe_key: str) -> pd.DataFrame:
        """
        【V110.6 终极流程修复版】根据配置为指定时间周期计算所有技术指标。
        - 核心修正 (本次修改):
          - [流程重构] 将 ma_convergence 和 zscore 的计算逻辑移至主计算循环中，
                        并确保它们在所有依赖项计算完毕后执行。
          - [映射修复] 将 ma_convergence 重新添加回 indicator_method_map。
        - 收益: 彻底解决了所有指标的计算顺序和依赖问题，流程完全理顺。
        """
        print(f"  [指标计算V110.6] 开始为周期 '{timeframe_key}' 计算指标...")
        if not config:
            print(f"    - 警告: 周期 '{timeframe_key}' 没有配置任何指标。")
            return df
        max_required_period = self._get_max_period_for_timeframe(config, timeframe_key)
        if len(df) < max_required_period:
            logger.warning(f"数据行数 ({len(df)}) 不足以满足周期 '{timeframe_key}' 的最大计算要求 ({max_required_period})，将跳过该周期的所有指标计算。")
            return df
        df_for_calc = df.copy()
        if 'close' in df_for_calc.columns:
            df_for_calc['pct_change'] = df_for_calc['close'].pct_change()
        indicator_method_map = {
            'ema': self.calculator.calculate_ema, 'vol_ma': self.calculator.calculate_vol_ma, 'trix': self.calculator.calculate_trix,
            'coppock': self.calculator.calculate_coppock, 'rsi': self.calculator.calculate_rsi, 'macd': self.calculator.calculate_macd,
            'dmi': self.calculator.calculate_dmi, 'roc': self.calculator.calculate_roc, 'boll_bands_and_width': self.calculator.calculate_boll_bands_and_width,
            'cmf': self.calculator.calculate_cmf, 'bias': self.calculator.calculate_bias, 'atrn': self.calculator.calculate_atrn,
            'atrr': self.calculator.calculate_atrr, 'obv': self.calculator.calculate_obv, 'kdj': self.calculator.calculate_kdj,
            'uo': self.calculator.calculate_uo, 'vwap': self.calculator.calculate_vwap, 'atr': self.calculator.calculate_atr,
            'consolidation_period': self.calculator.calculate_consolidation_period,
            'fibonacci_levels': self.calculator.calculate_fibonacci_levels,
            'price_volume_ma_comparison': self.calculator.calculate_price_volume_ma_comparison,
            # 将 ma_convergence 重新添加回映射
            'ma_convergence': self.calculator.calculate_ma_convergence,
        }
        def merge_results(result_data, target_df):
            if result_data is None or result_data.empty: return
            if isinstance(result_data, pd.Series):
                result_data = result_data.to_frame()
            if isinstance(result_data, pd.DataFrame):
                for col in result_data.columns:
                    target_df[col] = result_data[col]
            else:
                logger.warning(f"指标计算返回了未知类型 {type(result_data)}，已跳过。")
        
        # --- 阶段一: 在统一的无后缀命名空间下，完成所有指标计算 ---
        for indicator_key, params in config.items():
            indicator_name = indicator_key.lower()
            if timeframe_key == 'W' and indicator_name in ['cmf', 'rsi']: continue
            #简化跳过逻辑
            if indicator_name in ['说明', 'index_sync', 'cyq_perf'] or not params.get('enabled', False): continue
            if indicator_name not in indicator_method_map and indicator_name != 'zscore': # zscore 特殊处理
                logger.warning(f"    - 警告: 未找到指标 '{indicator_name}' 的计算方法，已跳过。")
                continue
            
            #将 Z-Score 的计算逻辑移到这里，确保它在依赖项之后执行
            if indicator_name == 'zscore':
                for z_config in params.get('configs', []):
                    if timeframe_key not in z_config.get("apply_on", []): continue
                    try:
                        source_pattern = z_config.get("source_column_pattern")
                        output_col_name = z_config.get("output_column_name")
                        window = z_config.get("window", 60)
                        source_col_name = source_pattern
                        if "{fast}" in source_pattern:
                            macd_cfg = config.get('macd', {})
                            macd_periods = next((c.get('periods') for c in macd_cfg.get('configs', []) if timeframe_key in c.get('apply_on', [])), None)
                            if macd_periods:
                                source_col_name = source_pattern.format(fast=macd_periods[0], slow=macd_periods[1], signal=macd_periods[2])
                            else: continue
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
                continue # 处理完Z-score后继续下一个指标

            # 统一处理所有其他指标
            configs_to_process = params.get('configs', [params])
            for sub_config in configs_to_process:
                if timeframe_key not in sub_config.get("apply_on", []): continue
                try:
                    method_to_call = indicator_method_map[indicator_name]
                    if indicator_name in ['consolidation_period', 'fibonacci_levels', 'price_volume_ma_comparison', 'ma_convergence']:
                        result_df = await method_to_call(df=df_for_calc, params=params)
                        merge_results(result_df, df_for_calc)
                        break # 复合指标只计算一次
                    
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
                            kwargs_iter['period'] = p_set[0] if isinstance(p_set, list) else p_set
                        result_df = await method_to_call(**kwargs_iter)
                        merge_results(result_df, df_for_calc)
                except Exception as e:
                    logger.error(f"    - 计算指标 {indicator_name.upper()} (周期: {timeframe_key}, 参数: {sub_config.get('periods')}) 时出错: {e}", exc_info=True)

        # --- 阶段二: 统一添加后缀并返回 ---
        suffix = f"_{timeframe_key}"
        rename_map = {
            col: f"{col}{suffix}"
            for col in df_for_calc.columns
            if not str(col).endswith(suffix)
        }
        final_df = df_for_calc.rename(columns=rename_map)
        return final_df

    async def _calculate_all_slopes(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V3.0 混合计算版】
        - 核心升级: 实现智能跳过逻辑。在计算斜率前，会检查目标列是否已存在。
                    如果列已存在（意味着它是由预计算任务加载并适配的），则跳过计算。
                    这使得本方法能无缝处理“预计算的筹码斜率”和“即时计算的其他指标斜率”。
        """
        # print("    - [斜率中心 V3.0 混合计算版] 启动...")
        slope_params = config.get('feature_engineering_params', {}).get('slope_params', {})
        if not slope_params.get('enabled', False):
            return all_dfs
        series_to_slope = slope_params.get('series_to_slope', {})
        if not series_to_slope:
            return all_dfs
        for col_pattern, lookbacks in series_to_slope.items():
            if "说明" in col_pattern: continue
            try:
                timeframe = col_pattern.split('_')[-1]
                if timeframe.upper() not in ['D', 'W', 'M'] and not timeframe.isdigit():
                    timeframe = 'D' # 默认日线
            except IndexError:
                continue
            if timeframe not in all_dfs or all_dfs[timeframe] is None:
                continue
            df = all_dfs[timeframe]
            if col_pattern not in df.columns:
                # print(f"      -> [跳过] 基础指标 '{col_pattern}' 在周期 '{timeframe}' 的DataFrame中不存在。")
                continue
            source_series = df[col_pattern].astype(float)
            for lookback in lookbacks:
                slope_col_name = f'SLOPE_{lookback}_{col_pattern}'
                # 智能检查逻辑：如果斜率列已存在，则跳过计算
                if slope_col_name in df.columns:
                    # print(f"      -> [跳过] 斜率 '{slope_col_name}' 已存在 (来自预计算).")
                    continue
                # print(f"      -> [即时计算] 正在生成斜率: '{slope_col_name}'...")
                min_p = max(2, lookback // 2)
                linreg_result = df.ta.linreg(close=source_series, length=lookback, min_periods=min_p, slope=True, intercept=False, r=False)
                slope_series = linreg_result if isinstance(linreg_result, pd.Series) else linreg_result.iloc[:, 0]
                df[slope_col_name] = slope_series.fillna(0)
            all_dfs[timeframe] = df

        # print("    - [斜率中心 V3.0] 所有斜率计算完成。")
        return all_dfs

    async def _calculate_all_accelerations(self, all_dfs: Dict[str, pd.DataFrame], config: Dict) -> Dict[str, pd.DataFrame]:
        """
        【V2.0 混合计算版】
        - 核心升级: 与斜率计算类似，增加了智能跳过逻辑。
                    如果目标加速度列已存在（来自预计算），则跳过，否则进行即时计算。
        """
        accel_params = config.get('feature_engineering_params', {}).get('accel_params', {})
        if not accel_params.get('enabled', False):
            return all_dfs
        series_to_accel = accel_params.get('series_to_accel', {})
        if not series_to_accel:
            return all_dfs
        # print("    - [加速度引擎 V2.0 混合计算版] 启动...")
        for base_col_name, periods in series_to_accel.items():
            if "说明" in base_col_name: continue
            timeframe = base_col_name.split('_')[-1]
            if timeframe not in all_dfs or all_dfs[timeframe] is None:
                continue
            df = all_dfs[timeframe]
            for period in periods:
                slope_col_name = f'SLOPE_{period}_{base_col_name}'
                if slope_col_name not in df.columns:
                    # print(f"      -> [跳过] 无法计算加速度，源斜率列 '{slope_col_name}' 不存在。")
                    continue
                accel_col_name = f'ACCEL_{period}_{base_col_name}'
                # 智能检查逻辑：如果加速度列已存在，则跳过计算
                if accel_col_name in df.columns:
                    # print(f"      -> [跳过] 加速度 '{accel_col_name}' 已存在 (来自预计算).")
                    continue
                # print(f"      -> [即时计算] 正在生成加速度: '{accel_col_name}'...")
                source_series = df[slope_col_name]
                min_p = max(2, period // 2)
                accel_linreg_result = df.ta.linreg(close=source_series, length=period, min_periods=min_p, slope=True, intercept=False, r=False)
                accel_series = accel_linreg_result if isinstance(accel_linreg_result, pd.Series) else accel_linreg_result.iloc[:, 0]
                df[accel_col_name] = accel_series.fillna(0)
        # print("    - [加速度引擎 V2.0] 计算完成。")
        return all_dfs

    async def _calculate_vpa_features(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V1.0 新增】VPA效率指标生产线
        - 核心职责: 计算全新的自定义指标 VPA_EFFICIENCY_D (资金攻击效率)。
        - 计算公式: 当日涨幅 / (当日成交量 / 21日均量)
        - 意义: 衡量每一单位的相对成交量，能换来多大的价格涨幅。
                 数值极低时，是典型的“天量滞涨”危险信号。
        """
        # print("    - [VPA效率生产线 V1.0 @ IndicatorService] 启动...")
        timeframe = 'D' # VPA效率是一个日线级别的概念
        if timeframe not in all_dfs:
            return all_dfs
        
        df = all_dfs[timeframe]
        
        # --- 1. 军备检查 ---
        required_cols = ['pct_change_D', 'volume_D', 'VOL_MA_21_D']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"      -> [严重警告] VPA效率生产线缺少关键数据: {missing}，模块已跳过！")
            return all_dfs

        # --- 2. 计算相对成交量倍数 ---
        # 为防止除以0的错误，将0替换为NaN，后续计算结果也会是NaN，最后统一填充为0
        volume_ratio = df['volume_D'] / df['VOL_MA_21_D'].replace(0, np.nan)

        # --- 3. 计算VPA效率 ---
        # 再次防止除以0
        vpa_efficiency = df['pct_change_D'] / volume_ratio.replace(0, np.nan)
        
        # 将新指标添加到DataFrame中，并将计算过程中可能产生的NaN和inf填充为0
        df['VPA_EFFICIENCY_D'] = vpa_efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        all_dfs[timeframe] = df
        # print("    - [VPA效率生产线 V1.0 @ IndicatorService] “资金攻击效率”指标生产完成。")
        return all_dfs

    async def _calculate_meta_features(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V2.0 赫斯特指数归位版】元特征计算车间
        - 核心职责: 在基础指标计算完毕后，专门计算那些依赖原始数据进行复杂
                    滚窗计算的“元指标”，如赫斯特指数、变异系数等。
        - 执行时机: 在基础指标计算之后，在斜率计算之前。
        - 核心修正: 将赫斯特指数的计算逻辑从 IntelligenceLayer 正确迁移至此，确保“计算与诊断分离”。
        """
        # print("    - [元特征车间 V2.0] 启动，正在计算赫斯特指数、CV等复杂特征...")
        
        # 我们主要关心日线数据
        timeframe = 'D'
        if timeframe not in all_dfs:
            return all_dfs
            
        df = all_dfs[timeframe]
        
        # --- 1. 计算赫斯特指数 (Hurst Exponent) ---
        # 此处是赫斯特指数计算的正确位置，保证其在斜率计算前完成。
        hurst_window = 120 # 暂时硬编码，未来可从配置读取
        hurst_col = f'hurst_{hurst_window}d_D'
        if 'close_D' in df.columns and hurst_col not in df.columns:
            try:
                # 【核心防御代码】确保计算的健壮性
                # 1. 创建一个没有NaN的副本用于计算，防止因数据空洞导致计算失败
                close_series_for_hurst = df['close_D'].dropna()
                
                # 2. 检查处理后数据是否仍然充足
                if len(close_series_for_hurst) < hurst_window:
                    print(f"      -> [警告] 去除NaN后，数据量({len(close_series_for_hurst)})不足以计算{hurst_window}周期的赫斯特指数，跳过。")
                    df[hurst_col] = np.nan # 将整列设为NaN，保证列存在
                else:
                    # 3. 在干净的数据上进行滚动计算
                    # math_tools.py 中的 hurst_exponent 函数已经足够健壮
                    hurst_values = close_series_for_hurst.rolling(hurst_window).apply(hurst_exponent, raw=True)
                    
                    # 4. 将计算结果对齐回原始的DataFrame索引，这是关键一步
                    df[hurst_col] = hurst_values.reindex(df.index)
                    # print(f"      -> 赫斯特指数 ({hurst_col}) 计算完成。")

            except Exception as e:
                print(f"      -> [严重警告] 赫斯特指数计算过程中发生未知错误: {e}")
                df[hurst_col] = np.nan # 确保即使出错，列也存在，防止下游流程报错

        # --- 2. 计算价格变异系数 (Price CV) ---
        cv_window = 60
        cv_col = f'price_cv_{cv_window}d_D'
        if 'close_D' in df.columns and cv_col not in df.columns:
            price_mean = df['close_D'].rolling(cv_window).mean()
            price_std = df['close_D'].rolling(cv_window).std()
            # 加上一个极小值防止除以零
            df[cv_col] = price_std / (price_mean + 1e-6)
            # print(f"      -> 价格变异系数 ({cv_col}) 计算完成。")

        # --- 3. 计算日线结构势能 (Energy Ratio) ---
        # 注意：这需要筹码数据已经合并到df中
        energy_col = 'energy_ratio_D'
        if 'support_below_D' in df.columns and 'pressure_above_D' in df.columns and energy_col not in df.columns:
            # 加上一个极小值防止除以零
            df[energy_col] = df['support_below_D'] / (df['pressure_above_D'] + 1e-6)
            # print(f"      -> 结构势能 ({energy_col}) 计算完成。")

        all_dfs[timeframe] = df
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
        
        # print(f"--- [IndustryService V2.0] {trade_date} 的行业结构化强度计算完成。 ---")
        return df.sort_values('strength_rank', ascending=False).set_index('industry_code')

    async def _process_single_industry_strength(self, industry, trade_date: datetime.date, market_daily_df: pd.DataFrame) -> Optional[Dict]:
        """
        处理单个行业的强度计算，便于并行化。
        """
        # print(f"  - 正在处理行业: {industry.name} ({industry.ts_code})")

        # 2. 并行获取该行业所需的所有数据
        start_date = trade_date - datetime.timedelta(days=self.momentum_lookback + 30)

        try:
            # 使用 asyncio.gather 获取一个行业的所有数据
            data_tasks = {
                "daily": self.indicator_dao.get_industry_daily_data(industry.ts_code, start_date, trade_date),
                "flow": self.fund_flow_dao.get_industry_fund_flow(industry.ts_code, start_date, trade_date)
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
                15 * rs_score +                # 相对强度分
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
        计算龙头效应得分。
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
        计算板块协同性（上涨广度）得分。
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
            
        # print(f"      - [协同性] 上涨家数/总数: {rising_count}/{total_count} (占比: {rising_ratio:.2%}), 大涨家数: {strong_rising_count}，得分: {score}")
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
            
        # print(f"      - [涨停梯队] 发现 {limit_up_count} 家涨停，得分: {score}")
        return score

    async def _calculate_relative_strength_score(self, industry_daily_df: pd.DataFrame, market_daily_df: pd.DataFrame) -> float:
        """
        计算行业相对大盘的强度得分。
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
        分析行业轮动，识别强度排名持续上升的板块。
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
