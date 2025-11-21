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
from .feature_engineering_service import FeatureEngineeringService
from .contextual_analysis_service import ContextualAnalysisService

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
        # 专业服务层
        self.calculator = IndicatorCalculator()
        self.feature_service = FeatureEngineeringService() # 实例化特征工程师
        self.context_service = ContextualAnalysisService(cache_manager_instance) # 实例化情报分析师
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
    def _rename_precomputed_derivatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V4.0 · 终极重构版】预计算衍生指标列名适配器
        - 核心重构: 废弃了原先脆弱且易错的 if/elif 链式匹配逻辑。
        - 解决方案: 采用单一、更强大的正则表达式，通过可选捕获组 `(...)?` 一次性、
                    无歧义地解析所有类型的衍生指标（包括常规、带sum的、斜率、加速度）。
                    这从根本上解决了因匹配顺序错误导致列名解析失败的BUG。
        """
        import re
        rename_map = {}
        # 定义一个能处理所有情况的、统一的正则表达式
        # 模式解释:
        #   (.+?)                     - 非贪婪匹配基础指标名 (base_name)
        #   (?:_sum_(\d+)d)?          - 可选的sum部分，捕获sum周期 (sum_period)
        #   _(slope|accel)_(\d+)d$    - 必须匹配的斜率/加速度部分，捕获类型(deriv_type)和周期(deriv_period)
        pattern = re.compile(r'(.+?)(?:_sum_(\d+)d)?_(slope|accel)_(\d+)d$')
        for col in df.columns:
            match = pattern.match(col)
            if match:
                base_name, sum_period, deriv_type, deriv_period = match.groups()
                # 根据捕获组构建新的、标准化的列名
                if sum_period:
                    # 处理带 _sum_ 的情况
                    # 例如: net_lg_amount_consensus_sum_13d_slope_13d -> SLOPE_13_net_lg_amount_consensus_sum_13d
                    new_name = f"{deriv_type.upper()}_{deriv_period}_{base_name}_sum_{sum_period}d"
                else:
                    # 处理常规情况
                    # 例如: RSI_13_slope_5d -> SLOPE_5_RSI_13
                    new_name = f"{deriv_type.upper()}_{deriv_period}_{base_name}"
                rename_map[col] = new_name
            elif col.endswith('_D'):
                # 保留对仅带 _D 后缀的列的处理逻辑
                rename_map[col] = col[:-2]
        if rename_map:
            df_renamed = df.rename(columns=rename_map)
            return df_renamed
        else:
            return df
    def _get_max_lookback_period(self, config: dict) -> int:
        """
        【军需官】扫描整个策略配置，找出所有指标中要求的最长回溯期。
        这是一个简化的实现，用于演示核心思想。
        """
        # print("    - [军需官] 正在扫描全军军火库，确定最大回溯需求...")
        # 简化实现：
        calculated_max = 350 # 保守估计，足以满足EMA(55周)等大周期指标
        # print(f"    - [军需官] 扫描完成，最大回溯需求估算为 {calculated_max} 个日线周期。")
        return calculated_max
    async def prepare_data_for_strategy(
        self,
        stock_code: str,
        config: dict,
        trade_time: Optional[str] = None,
        latest_only: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        【V8.3 · 军械库清单生成版】为策略准备数据的统一入口。
        - 核心修复: 调整了特征计算的顺序，确保 `breakout_quality_score` 在其所有依赖项（如VPA_EFFICIENCY）计算完毕后才执行，从根本上解决了流程错乱问题。
        - 【新增】在所有数据准备和计算流程结束后，调用 `_log_final_data_columns` 输出最终的数据清单。
        """
        # --- 步骤 1: 【第一道工序】准备基础数据和常规指标 ---
        all_dfs = await self._prepare_base_data_and_indicators(stock_code, config, trade_time, latest_only=latest_only)
        if not all_dfs:
            return {}
        indicators_config = config.get('feature_engineering_params', {}).get('indicators', {})
        # --- 步骤 2: 【形态增强信号计算】 ---
        all_dfs = await self.feature_service.calculate_pattern_enhancement_signals(all_dfs, config, self.calculator)
        # --- 步骤 3: 【VPA效率指标计算】 ---
        all_dfs = await self.feature_service.calculate_vpa_features(all_dfs, config)
        # --- 步骤 4: 【突破质量分计算】(核心修正：移至此处，确保依赖项已就绪) ---
        bqs_params = indicators_config.get('breakout_quality_score', {})
        if bqs_params.get('enabled', False):
            all_dfs = await self.feature_service.calculate_breakout_quality(all_dfs, bqs_params, self.calculator)
        # --- 5. 【元特征计算】 ---
        all_dfs = await self.feature_service.calculate_meta_features(all_dfs, config)
        # --- 6. 【均线势能计算】 ---
        ma_potential_params = indicators_config.get('ma_potential_metrics', {})
        if ma_potential_params.get('enabled', False):
            all_dfs = await self.feature_service.calculate_ma_potential_metrics(all_dfs, ma_potential_params)
        # --- 7. 【盘整期计算】 ---
        consolidation_params = indicators_config.get('consolidation_period', {})
        if consolidation_params.get('enabled', False):
            all_dfs = await self.feature_service.calculate_consolidation_period(all_dfs, consolidation_params)
        # --- 8. 【斜率与加速度计算】 ---
        all_dfs = await self.feature_service.calculate_all_slopes(all_dfs, config)
        all_dfs = await self.feature_service.calculate_all_accelerations(all_dfs, config)
        # --- 9. 【高级模式识别】 ---
        all_dfs = await self.feature_service.calculate_pattern_recognition_signals(all_dfs, config)
        # --- 10. 【上下文信息注入】 ---
        if not all_dfs or 'D' not in all_dfs or all_dfs['D'].empty:
            return all_dfs
        df_daily = all_dfs['D']
        start_date = df_daily.index.min().date()
        end_date = df_daily.index.max().date()
        # (后续的上下文注入逻辑保持不变)
        hot_money_params = self._find_params_recursively(config, 'hot_money_params')
        if hot_money_params and hot_money_params.get('enabled', False):
            hm_signals_df = await self.context_service.prepare_hot_money_signals(stock_code, start_date, end_date, hot_money_params)
            if not hm_signals_df.empty:
                df_daily = df_daily.merge(hm_signals_df, left_index=True, right_index=True, how='left')
                for col in hm_signals_df.columns: df_daily[col] = df_daily[col].fillna(False).astype(bool)
        sentiment_params = self._find_params_recursively(config, 'market_sentiment_params')
        if sentiment_params and sentiment_params.get('enabled', False):
            sentiment_signals_df = await self.context_service.prepare_market_sentiment_signals(stock_code, start_date, end_date, sentiment_params)
            if not sentiment_signals_df.empty:
                sentiment_signals_df.index = pd.to_datetime(sentiment_signals_df.index, utc=True)
                df_daily = df_daily.merge(sentiment_signals_df, left_index=True, right_index=True, how='left')
                for col in sentiment_signals_df.columns:
                    fill_value = 0 if 'score' in col or 'rank' in col or 'ups' in col else False
                    df_daily[col] = df_daily[col].fillna(fill_value)
        four_layer_params = self._find_params_recursively(config, 'four_layer_scoring_params')
        industry_params = four_layer_params.get('industry_lifecycle_scoring_params', {}) if four_layer_params else {}
        if industry_params and industry_params.get('enabled', False):
            industry_lifecycle_df = await self.context_service.prepare_fused_industry_signals(
                stock_code, start_date, end_date, industry_params
            )
            if not industry_lifecycle_df.empty:
                df_daily = df_daily.merge(industry_lifecycle_df, left_index=True, right_index=True, how='left')
                for col in industry_lifecycle_df.columns:
                    df_daily[col] = df_daily[col].ffill().fillna(0)
        kpl_params = self._find_params_recursively(config, 'kpl_theme_params')
        if kpl_params and kpl_params.get('enabled', False):
            kpl_hotness_df = await self.context_service.analyze_kpl_theme_hotness(
                stock_code, start_date, end_date, kpl_params
            )
            if not kpl_hotness_df.empty:
                kpl_hotness_df.index = pd.to_datetime(kpl_hotness_df.index, utc=True)
                df_daily = df_daily.merge(kpl_hotness_df, left_index=True, right_index=True, how='left')
                df_daily['THEME_HOTNESS_SCORE_D'].fillna(0, inplace=True)
            else:
                df_daily['THEME_HOTNESS_SCORE_D'] = 0.0
        smart_money_params = self._find_params_recursively(config, 'smart_money_params')
        if smart_money_params and smart_money_params.get('enabled', False):
            smart_money_signals_df = await self.context_service.prepare_smart_money_signals(stock_code, start_date, end_date, smart_money_params)
            if not smart_money_signals_df.empty:
                smart_money_signals_df.index = pd.to_datetime(smart_money_signals_df.index, utc=True)
                df_daily = df_daily.merge(smart_money_signals_df, left_index=True, right_index=True, how='left')
                for col in smart_money_signals_df.columns:
                    df_daily[col] = df_daily[col].fillna(False).astype(bool)
        all_dfs['D'] = df_daily
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
        【V8.21 · 资金流列名标准化与NMFNF/OCH计算版】
        - 核心修复: 修正了 `fund_flow_tushare` 数据源的列名标准化逻辑，确保其原始买卖金额列在合并到 `df_daily_master` 之前，
                      被正确地命名为带有 `_D` 后缀的形式，解决了 `_diagnose_axiom_consensus` 方法中缺少 `net_sm_amount_calibrated_D` 等列的问题。
        - 核心修复: 调整了特征计算的顺序，确保 `breakout_quality_score` 在其所有依赖项（如VPA_EFFICIENCY）计算完毕后才执行，从根本上解决了流程错乱问题。
        - 【新增】在所有数据准备和计算流程结束后，调用 `_log_final_data_columns` 输出最终的数据清单。
        - 【新增】在日线数据准备完成后，计算 `NMFNF_D` 和 `OCH_D`。
        """
        required_tfs = self._discover_required_timeframes_from_config(config)
        pattern_enhancement_params = config.get('feature_engineering_params', {}).get('indicators', {}).get('pattern_enhancement_signals', {})
        if pattern_enhancement_params.get('enabled', False):
            minute_tf = pattern_enhancement_params.get('minute_level_tf')
            if minute_tf:
                required_tfs.add(minute_tf)
                logger.info(f"检测到形态增强信号已启用，已将分钟周期 '{minute_tf}' 加入数据获取计划。")
        if not required_tfs:
            print("    - [配置读取] 未发现任何需要的时间周期，处理终止。")
            return {}
        if latest_only:
            max_lookback = self._get_max_lookback_period(config)
            safety_buffer = 100
            base_needed_bars = max_lookback + safety_buffer
            print(f"    - [闪电模式启动] 策略最大回溯期: {max_lookback}, 安全缓冲: {safety_buffer}, 最终加载: {base_needed_bars} 条记录。")
        else:
            base_needed_bars = config.get('feature_engineering_params', {}).get('base_needed_bars', 1200)
        base_tfs_to_fetch = set()
        resample_map = {}
        for tf in required_tfs:
            if tf in ['W', 'M']:
                base_tfs_to_fetch.add('D')
                resample_map[tf] = 'D'
            else:
                base_tfs_to_fetch.add(tf)
        indicators_config = config.get('feature_engineering_params', {}).get('indicators', {})
        tasks = []
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
                return ('legacy_supplemental', df)
            tasks.append(_fetch_legacy_supplemental_tagged(stock_code, trade_time, base_needed_bars))
        async def _fetch_advanced_chips_tagged(stock_code, trade_time, limit):
            trade_time_dt = pd.to_datetime(trade_time, utc=True) if trade_time else None
            df = await self.strategies_dao.get_advanced_chip_metrics_data(stock_code, trade_time_dt, limit)
            return ('advanced_chips', df)
        tasks.append(_fetch_advanced_chips_tagged(stock_code, trade_time, base_needed_bars))
        async def _fetch_daily_basic_tagged(stock_code, trade_time, limit):
            trade_time_dt = pd.to_datetime(trade_time, utc=True) if trade_time else None
            df = await self.strategies_dao.get_daily_basic_data(stock_code, trade_time_dt, limit)
            return ('daily_basic', df)
        tasks.append(_fetch_daily_basic_tagged(stock_code, trade_time, base_needed_bars))
        trade_time_dt_date = pd.to_datetime(trade_time, utc=True).date() if trade_time else datetime.datetime.now().date()
        async def _fetch_fund_flow_ths_tagged(stock_code, trade_time_dt_date, limit):
            df = await self.fund_flow_dao.get_fund_flow_ths_data(stock_code, trade_time_dt_date, limit)
            return ('fund_flow_ths', df)
        tasks.append(_fetch_fund_flow_ths_tagged(stock_code, trade_time_dt_date, base_needed_bars))
        async def _fetch_fund_flow_dc_tagged(stock_code, trade_time_dt_date, limit):
            df = await self.fund_flow_dao.get_fund_flow_dc_data(stock_code, trade_time_dt_date, limit)
            return ('fund_flow_dc', df)
        tasks.append(_fetch_fund_flow_dc_tagged(stock_code, trade_time_dt_date, base_needed_bars))
        async def _fetch_fund_flow_tushare_tagged(stock_code, trade_time_dt_date, limit):
            df = await self.fund_flow_dao.get_fund_flow_daily_data(stock_code, trade_time_dt_date, limit)
            # 对 fund_flow_tushare 的列名进行标准化，添加 _D 后缀
            if not df.empty:
                rename_map = {col: f"{col}_D" for col in df.columns if col not in ['trade_time', 'ts_code']}
                df.rename(columns=rename_map, inplace=True)
            return ('fund_flow_tushare', df)
        tasks.append(_fetch_fund_flow_tushare_tagged(stock_code, trade_time_dt_date, base_needed_bars))
        async def _fetch_advanced_fund_flow_tagged(stock_code, trade_time_dt_date, limit):
            df = await self.fund_flow_dao.get_advanced_fund_flow_metrics_data(stock_code, trade_time_dt_date, limit)
            return ('advanced_fund_flow', df)
        tasks.append(_fetch_advanced_fund_flow_tagged(stock_code, trade_time_dt_date, base_needed_bars))
        async def _fetch_price_limit_tagged(stock_code, trade_time, limit):
            trade_time_dt = pd.to_datetime(trade_time, utc=True).date() if trade_time else None
            df = await self.stock_trade_dao.get_price_limit_data(stock_code, trade_time_dt, limit)
            return ('price_limit', df)
        tasks.append(_fetch_price_limit_tagged(stock_code, trade_time, base_needed_bars))
        async def _fetch_and_tag_data(tf_to_fetch, trade_time_str):
            df = await self._get_ohlcv_data(stock_code, tf_to_fetch, base_needed_bars, trade_time_str)
            return (tf_to_fetch, df)
        for tf in base_tfs_to_fetch:
            tasks.append(_fetch_and_tag_data(tf, trade_time))
        all_data_results = await asyncio.gather(*tasks, return_exceptions=True)
        raw_dfs: Dict[str, pd.DataFrame] = {}
        supplemental_dfs: Dict[str, pd.DataFrame] = {}
        for result in all_data_results:
            if isinstance(result, Exception):
                print(f"      -> 警告: 一个数据获取任务失败: {result}")
                continue
            if not (isinstance(result, tuple) and len(result) == 2): continue
            tag, data = result
            if isinstance(data, pd.DataFrame) and not data.empty:
                object_cols = data.select_dtypes(include=['object']).columns
                for col in object_cols:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                if tag in ['legacy_supplemental', 'advanced_chips', 'daily_basic', 'fund_flow_ths', 'fund_flow_dc', 'fund_flow_tushare', 'advanced_fund_flow', 'price_limit']:
                    supplemental_dfs[tag] = data
                else:
                    raw_dfs[tag] = data
        if 'D' not in raw_dfs:
            print(f"    - 错误: 最核心的日线数据获取失败，处理终止。")
            return {}
        df_daily_master = raw_dfs['D']
        df_daily_master.index = df_daily_master.index.normalize()
        processed_supp_dfs_to_join = []
        all_new_cols = []
        for tag, df_supp in supplemental_dfs.items():
            df_supp_std = self._standardize_df_index_to_utc(df_supp)
            if df_supp_std is None or df_supp_std.empty:
                continue
            df_supp_std.index = df_supp_std.index.normalize()
            if tag in ['advanced_chips', 'advanced_fund_flow']:
                df_supp_std = self._rename_precomputed_derivatives(df_supp_std)
            if tag == 'daily_basic':
                # 确保 total_mv 被正确重命名为 total_market_value_D
                if 'total_mv' in df_supp_std.columns:
                    df_supp_std.rename(columns={'total_mv': 'total_market_value'}, inplace=True)
                # 确保所有 daily_basic 的列都带上 _D 后缀
                df_supp_std = df_supp_std.add_suffix('_D')
            # 针对 fund_flow_tushare，不再添加额外的后缀，因为其列名已在 _fetch_fund_flow_tushare_tagged 中标准化为 _D
            if tag in ['fund_flow_ths', 'fund_flow_dc']: # fund_flow_tushare 已在获取时处理
                suffix = f"_{tag}"
                df_supp_std = df_supp_std.add_suffix(suffix)
            else:
                # 避免与 df_daily_master 现有列冲突，但要保留 total_market_value_D
                conflicting_cols = df_daily_master.columns.intersection(df_supp_std.columns)
                # 排除 total_market_value_D，因为它可能在 df_supp_std 中被正确命名
                if 'total_market_value_D' in conflicting_cols:
                    conflicting_cols = conflicting_cols.drop('total_market_value_D')
                if not conflicting_cols.empty:
                    df_supp_std = df_supp_std.drop(columns=conflicting_cols)
            if not df_supp_std.columns.empty:
                processed_supp_dfs_to_join.append(df_supp_std)
                all_new_cols.extend(df_supp_std.columns)
        if processed_supp_dfs_to_join:
            df_daily_master = df_daily_master.join(processed_supp_dfs_to_join, how='left')
            unique_new_cols = list(set(all_new_cols))
            cols_to_ffill = [col for col in unique_new_cols if col in df_daily_master.columns]
            if cols_to_ffill:
                df_daily_master[cols_to_ffill] = df_daily_master[cols_to_ffill].ffill()
        rename_map_daily = {
            col: f"{col}_D"
            for col in df_daily_master.columns
            if not col.endswith(('_D', '_W', '_M')) and not any(col.endswith(f'_{i}') for i in range(100))
        }
        df_daily_master.rename(columns=rename_map_daily, inplace=True)
        if 'close_D' in df_daily_master.columns:
            df_daily_master['pct_change_D'] = df_daily_master['close_D'].pct_change().fillna(0)
        raw_dfs['D'] = df_daily_master
        # NEW STEP: Calculate AAA_D for the daily dataframe
        # This needs to happen before _calculate_indicators_for_timescale is called for 'D'
        # because DMA depends on AAA_D.
        if 'D' in raw_dfs and not raw_dfs['D'].empty:
            temp_dfs = {'D': raw_dfs['D']}
            temp_dfs = await self.feature_service.calculate_aaa_indicator(temp_dfs)
            raw_dfs['D'] = temp_dfs['D']
            # NEW STEP: Calculate NMFNF_D for the daily dataframe
            temp_dfs = {'D': raw_dfs['D']} # 重新封装，确保传入的是字典
            temp_dfs = await self.feature_service.calculate_nmfnf(temp_dfs)
            raw_dfs['D'] = temp_dfs['D']
            # NEW STEP: Calculate OCH_D for the daily dataframe
            temp_dfs = {'D': raw_dfs['D']} # 重新封装，确保传入的是字典
            temp_dfs = await self.feature_service.calculate_och(temp_dfs)
            raw_dfs['D'] = temp_dfs['D']
        if resample_map:
            df_daily = raw_dfs['D']
            for target_tf, source_tf in resample_map.items():
                if source_tf == 'D' and not df_daily.empty:
                    aggregation_rules = {
                        'open_D': 'first', 'high_D': 'max', 'low_D': 'min', 'close_D': 'last', 'volume_D': 'sum'
                    }
                    for col in df_daily.columns:
                        if col not in aggregation_rules:
                            if 'amount' in col.lower() or 'vol' in col.lower() or 'net' in col.lower() or 'flow' in col.lower() or 'value' in col.lower() or 'och' in col.lower() or 'nmfnf' in col.lower(): # 增加对资金流、市值、OCH、NMFNF列的聚合规则
                                aggregation_rules[col] = 'sum' if 'flow' in col.lower() or 'amount' in col.lower() else 'last' # 资金流和金额求和，其他取last
                            else:
                                aggregation_rules[col] = 'last'
                    if 'turnover_rate_D' in aggregation_rules:
                        aggregation_rules['turnover_rate_D'] = 'mean'
                    resample_period = 'W-FRI' if target_tf == 'W' else 'ME'
                    df_resampled = df_daily.resample(resample_period).agg(aggregation_rules)
                    df_resampled.dropna(how='all', inplace=True)
                    if not df_resampled.empty:
                        rename_map_resampled = {col: col.replace('_D', f'_{target_tf}') for col in df_resampled.columns if col.endswith('_D')}
                        df_resampled.rename(columns=rename_map_resampled, inplace=True)
                        if target_tf == 'W':
                            df_synthetic_indicators = self._calculate_synthetic_weekly_indicators(df_daily, df_resampled)
                            df_resampled = df_resampled.merge(df_synthetic_indicators, left_index=True, right_index=True, how='left')
                        raw_dfs[target_tf] = df_resampled
        processed_dfs: Dict[str, pd.DataFrame] = {}
        calc_tasks = []
        async def _calculate_for_tf(tf, df):
            df = self._standardize_df_index_to_utc(df)
            df_with_indicators = await self._calculate_indicators_for_timescale(df, indicators_config, tf)
            return tf, df_with_indicators
        for tf, df in raw_dfs.items():
            if tf in required_tfs:
                calc_tasks.append(_calculate_for_tf(tf, df))
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
        return processed_dfs
    def _calculate_synthetic_weekly_indicators(self, df_daily: pd.DataFrame, df_weekly: pd.DataFrame) -> pd.DataFrame:
        """
        【V8.3 · 命名协议同步版】高级指标合成室
        - 核心修复: 更新内部逻辑，使其使用带有 '_D' 后缀的日线列名进行计算，以适应上游的标准化流程。
        """
        synthetic_indicators = pd.DataFrame(index=df_weekly.index)
        # --- 1. 合成周线CMF (Chaikin Money Flow) ---
        # 使用带 '_D' 后缀的列名
        mfm = ((df_daily['close_D'] - df_daily['low_D']) - (df_daily['high_D'] - df_daily['close_D'])) / (df_daily['high_D'] - df_daily['low_D'])
        mfm = mfm.fillna(0)
        daily_mfv = mfm * df_daily['volume_D']
        weekly_mfv_sum = daily_mfv.resample('W-FRI').sum()
        weekly_volume_sum = df_daily['volume_D'].resample('W-FRI').sum()
        cmf_period = 21
        cmf_numerator = weekly_mfv_sum.rolling(window=cmf_period).sum()
        cmf_denominator = weekly_volume_sum.rolling(window=cmf_period).sum()
        synthetic_indicators['CMF_21_W'] = np.divide(cmf_numerator, cmf_denominator, out=np.full_like(cmf_numerator, np.nan), where=cmf_denominator!=0)
        # --- 2. 合成周线RSI (Relative Strength Index) ---
        # 使用带 '_D' 后缀的列名
        delta = df_daily['close_D'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        weekly_gain_sum = gain.resample('W-FRI').sum()
        weekly_loss_sum = loss.resample('W-FRI').sum()
        rsi_period = 13
        avg_gain = ta.rma(weekly_gain_sum, length=rsi_period)
        avg_loss = ta.rma(weekly_loss_sum, length=rsi_period)
        if avg_gain is None or avg_loss is None:
            print(f"调试信息: 合成周线RSI失败，因为 avg_gain 或 avg_loss 为 None。通常是由于周线数据不足 {rsi_period} 条所致。")
            synthetic_indicators['RSI_13_W'] = np.nan
        else:
            rs = avg_gain / (avg_loss + 1e-9) 
            rsi = 100 - (100 / (1 + rs))
            synthetic_indicators['RSI_13_W'] = rsi
        return synthetic_indicators
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
        【V110.20 · 通达信指标集成与列名修复版】根据配置为指定时间周期计算所有技术指标。
        - 核心修复: 调整了 `merge_results` 函数的逻辑，确保只对**不以时间框架后缀结尾**的列添加后缀，避免重复。
        """
        if not config:
            return df
        max_required_period = self._get_max_period_for_timeframe(config, timeframe_key)
        if len(df) < max_required_period:
            logger.warning(f"数据行数 ({len(df)}) 不足以满足周期 '{timeframe_key}' 的最大计算要求 ({max_required_period})，将跳过。")
            return df
        df_for_calc = df.copy()
        indicator_method_map = {
            'ma': self.calculator.calculate_ma,
            'ema': self.calculator.calculate_ema, 'vol_ma': self.calculator.calculate_vol_ma, 'trix': self.calculator.calculate_trix,
            'coppock': self.calculator.calculate_coppock, 'rsi': self.calculator.calculate_rsi, 'macd': self.calculator.calculate_macd,
            'dmi': self.calculator.calculate_dmi, 'roc': self.calculator.calculate_roc, 'boll_bands_and_width': self.calculator.calculate_boll_bands_and_width,
            'cmf': self.calculator.calculate_cmf, 'bias': self.calculator.calculate_bias, 'atrn': self.calculator.calculate_atrn,
            'atrr': self.calculator.calculate_atrr, 'obv': self.calculator.calculate_obv, 'kdj': self.calculator.calculate_kdj,
            'uo': self.calculator.calculate_uo, 'vwap': self.calculator.calculate_vwap, 'atr': self.calculator.calculate_atr,
            'fibonacci_levels': self.calculator.calculate_fibonacci_levels,
            'price_volume_ma_comparison': self.calculator.calculate_price_volume_ma_comparison,
            'dma': self.calculator.calculate_dma,
            'atan_ma_angle': self.calculator.calculate_atan_ma_angle,
            'ma_velocity_acceleration': self.calculator.calculate_ma_velocity_acceleration,
            'zigzag': self.calculator.calculate_zigzag,
        }
        def merge_results(result_data, target_df):
            if result_data is None or result_data.empty:
                return
            if isinstance(result_data, pd.Series):
                result_data = result_data.to_frame()
            if isinstance(result_data, pd.DataFrame):
                suffix = f"_{timeframe_key}"
                # 只有当列名不以当前时间框架后缀结尾时，才添加后缀
                rename_dict = {col: f"{col}{suffix}" for col in result_data.columns if not col.endswith(suffix)}
                result_data.rename(columns=rename_dict, inplace=True)
                for col in result_data.columns:
                    target_df[col] = result_data[col]
            else:
                logger.warning(f"指标计算返回了未知类型 {type(result_data)}，已跳过。")
        ordered_calc_keys = [
            'ma', 'ema', 'vol_ma', 'macd', 'dmi', 'rsi', 'roc', 'boll_bands_and_width', 'kdj', 'trix', 'coppock', 'cmf', 'bias', 'atr', 'obv', 'vwap', 'uo',
            'price_volume_ma_comparison', 'zscore',
            'fibonacci_levels', 'dma', 'atan_ma_angle', 'ma_velocity_acceleration', 'zigzag'
        ]
        close_col_tf = f'close_{timeframe_key}'
        high_col_tf = f'high_{timeframe_key}'
        low_col_tf = f'low_{timeframe_key}'
        open_col_tf = f'open_{timeframe_key}'
        volume_col_tf = f'volume_{timeframe_key}'
        amount_col_tf = f'amount_{timeframe_key}'
        for indicator_key in ordered_calc_keys:
            params = config.get(indicator_key)
            if not params or not params.get('enabled', False): continue
            indicator_name = indicator_key.lower()
            if timeframe_key == 'W' and indicator_name in ['cmf', 'rsi']: continue
            if indicator_name == 'zscore':
                for z_config in params.get('configs', []):
                    if timeframe_key not in z_config.get("apply_on", []): continue
                    try:
                        source_pattern = z_config.get("source_column_pattern")
                        output_col_name = z_config.get("output_column_name")
                        window = z_config.get("window", 60)
                        source_pattern_no_suffix = source_pattern.removesuffix(f"_{timeframe_key}")
                        macd_cfg = config.get('macd', {})
                        macd_periods = next((c.get('periods') for c in macd_cfg.get('configs', []) if timeframe_key in c.get('apply_on', [])), None)
                        if macd_periods:
                            source_col_name = source_pattern_no_suffix.format(fast=macd_periods[0], slow=macd_periods[1], signal=macd_periods[2]) + f"_{timeframe_key}"
                        else: continue
                        if source_col_name in df_for_calc.columns:
                            source_series = df_for_calc[source_col_name]
                            zscore_result = ((source_series - source_series.rolling(window=window).mean()) / source_series.rolling(window=window).std()).fillna(0)
                            df_for_calc[output_col_name.removesuffix(f"_{timeframe_key}") + f"_{timeframe_key}"] = zscore_result
                        else:
                            logger.warning(f"Z-score计算失败：源列 '{source_col_name}' 在临时DataFrame中不存在。")
                    except Exception as e:
                        logger.error(f"计算Z-score时出错: {e}", exc_info=True)
                continue
            configs_to_process = params.get('configs', [params])
            for sub_config in configs_to_process:
                if timeframe_key not in sub_config.get("apply_on", []): continue
                try:
                    method_to_call = indicator_method_map[indicator_name]
                    if indicator_name in ['fibonacci_levels', 'price_volume_ma_comparison']:
                        result_df = await method_to_call(df=df_for_calc, params=sub_config, suffix=f"_{timeframe_key}")
                        merge_results(result_df, df_for_calc)
                        continue
                    kwargs = {'df': df_for_calc}
                    periods = sub_config.get('periods')
                    if indicator_name == 'vwap':
                        kwargs.update({
                            'anchor': 'D' if timeframe_key.isdigit() else timeframe_key,
                            'high_col': high_col_tf, 'low_col': low_col_tf,
                            'close_col': close_col_tf, 'volume_col': volume_col_tf,
                            'suffix': f"_{timeframe_key}"
                        })
                        result_df = await method_to_call(**kwargs)
                        merge_results(result_df, df_for_calc)
                        continue
                    if indicator_name == 'dma':
                        smooth_factor_col = sub_config.get('smooth_factor_col')
                        if smooth_factor_col and smooth_factor_col in df_for_calc.columns:
                            kwargs.update({'smooth_factor_series': df_for_calc[smooth_factor_col], 'close_col': close_col_tf})
                            result_df = await method_to_call(**kwargs)
                            merge_results(result_df, df_for_calc)
                        else:
                            logger.warning(f"DMA计算失败：缺少平滑因子列 '{smooth_factor_col}'。")
                        continue
                    if indicator_name == 'atan_ma_angle':
                        ma_col_base = sub_config.get('ma_col')
                        if ma_col_base and f"{ma_col_base}_{timeframe_key}" in df_for_calc.columns:
                            # 传递 ma_col_base 和 timeframe_key，但不在 kwargs 中添加 suffix
                            kwargs.update({'ma_col_base': ma_col_base, 'timeframe_key': timeframe_key})
                            result_df = await method_to_call(**kwargs)
                            merge_results(result_df, df_for_calc)
                        else:
                            logger.warning(f"ATAN 均线角度计算失败：缺少均线列 '{ma_col_base}_{timeframe_key}'。")
                        continue
                    if indicator_name == 'ma_velocity_acceleration':
                        ma_col_base = sub_config.get('ma_col')
                        if ma_col_base and f"{ma_col_base}_{timeframe_key}" in df_for_calc.columns:
                            # 传递 ma_col_base 和 timeframe_key，但不在 kwargs 中添加 suffix
                            kwargs.update({'ma_col_base': ma_col_base, 'timeframe_key': timeframe_key, 'ema_period': sub_config.get('ema_period', 3), 'sma_period': sub_config.get('sma_period', 3)})
                            result_df = await method_to_call(**kwargs)
                            merge_results(result_df, df_for_calc)
                        else:
                            logger.warning(f"均线速度加速度计算失败：缺少均线列 '{ma_col_base}_{timeframe_key}'。")
                        continue
                    if indicator_name == 'zigzag':
                        kwargs.update({'period': sub_config.get('period', 3), 'percent': sub_config.get('percent', 5.0), 'close_col': close_col_tf})
                        result_df = await method_to_call(**kwargs)
                        merge_results(result_df, df_for_calc)
                        continue
                    if periods is None:
                        if indicator_name == 'obv':
                            kwargs.update({'close_col': close_col_tf, 'volume_col': volume_col_tf})
                        result_df = await method_to_call(**kwargs)
                        merge_results(result_df, df_for_calc)
                        continue
                    is_multi_param = indicator_name in ['macd', 'trix', 'coppock', 'kdj', 'uo']
                    is_nested_list = isinstance(periods[0], list) if periods else False
                    periods_to_iterate = [periods] if is_multi_param and not is_nested_list else periods
                    for p_set in periods_to_iterate:
                        kwargs_iter = {'df': df_for_calc}
                        if indicator_name in ['ma', 'ema', 'rsi', 'roc', 'bias', 'mom']:
                            kwargs_iter.update({'period': p_set, 'close_col': close_col_tf})
                        elif indicator_name == 'vol_ma':
                            kwargs_iter.update({'period': p_set, 'volume_col': volume_col_tf})
                        elif indicator_name == 'macd':
                            kwargs_iter.update({'period_fast': p_set[0], 'period_slow': p_set[1], 'signal_period': p_set[2], 'close_col': close_col_tf})
                        elif indicator_name == 'trix':
                            kwargs_iter.update({'period': p_set[0], 'signal_period': p_set[1], 'close_col': close_col_tf})
                        elif indicator_name == 'coppock':
                            kwargs_iter.update({'long_roc_period': p_set[0], 'short_roc_period': p_set[1], 'wma_period': p_set[2], 'close_col': close_col_tf})
                        elif indicator_name in ['dmi', 'kdj', 'atr', 'atrn', 'atrr']:
                             kwargs_iter.update({'period': p_set, 'high_col': high_col_tf, 'low_col': low_col_tf, 'close_col': close_col_tf})
                        elif indicator_name == 'boll_bands_and_width':
                            kwargs_iter.update({'period': p_set, 'std_dev': float(sub_config.get('std_dev', 2.0)), 'close_col': close_col_tf, 'suffix': f"_{timeframe_key}"})
                        elif indicator_name == 'cmf':
                            kwargs_iter.update({'period': p_set, 'high_col': high_col_tf, 'low_col': low_col_tf, 'close_col': close_col_tf, 'volume_col': volume_col_tf})
                        elif indicator_name == 'uo':
                            kwargs_iter.update({'short_period': p_set[0], 'medium_period': p_set[1], 'long_period': p_set[2], 'high_col': high_col_tf, 'low_col': low_col_tf, 'close_col': close_col_tf})
                        else:
                            kwargs_iter['period'] = p_set[0] if isinstance(p_set, list) else p_set
                        result_df = await method_to_call(**kwargs_iter)
                        merge_results(result_df, df_for_calc)
                except Exception as e:
                    logger.error(f"    - 计算指标 {indicator_name.upper()} (周期: {timeframe_key}) 时出错: {e}", exc_info=True)
        return df_for_calc
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









