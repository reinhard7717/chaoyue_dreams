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
from typing import List, Optional, Set, Union, Dict, Callable, Awaitable
import pandas_ta as ta
import re
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

def _get_authoritative_daily_cols() -> Set[str]:
    """
    动态生成所有应优先使用补充数据源的列名（已带 _D 后缀）。
    这些列通常是预计算的、更专业的指标，其值应覆盖基础数据中的同名列。
    """
    cols = set()

    # StockDailyBasic fields (from strategies_dao.get_daily_basic_data)
    # Note: total_mv is renamed to total_market_value in DAO
    for field in ['turnover_rate', 'turnover_rate_f', 'volume_ratio', 'pe_ttm', 'pb', 'total_market_value', 'circ_mv']:
        cols.add(f"{field}_D")
    
    # StockCyqPerf fields (from strategies_dao.get_fund_flow_and_chips_data, renamed in DAO)
    # Note: his_low, his_high, cost_5pct, cost_50pct, cost_95pct are also part of this data source
    for field in ['CYQ_cost_15pct', 'CYQ_cost_85pct', 'CYQ_weight_avg', 'CYQ_winner_rate', 'his_low', 'his_high', 'cost_5pct', 'cost_50pct', 'cost_95pct']:
        cols.add(f"{field}_D")

    # BaseAdvancedChipMetrics
    for name in BaseAdvancedChipMetrics.CORE_METRICS.keys():
        cols.add(f"{name}_D")
        if name not in BaseAdvancedChipMetrics.SLOPE_ACCEL_EXCLUSIONS:
            for p in BaseAdvancedChipMetrics.UNIFIED_PERIODS:
                cols.add(f"{name}_slope_{p}d_D")
                cols.add(f"{name}_accel_{p}d_D")

    # BaseAdvancedFundFlowMetrics
    for name in BaseAdvancedFundFlowMetrics.CORE_METRICS.keys():
        cols.add(f"{name}_D")
        if name not in BaseAdvancedFundFlowMetrics.SLOPE_ACCEL_EXCLUSIONS:
            for p in BaseAdvancedFundFlowMetrics.UNIFIED_PERIODS: # Assuming same unified periods
                cols.add(f"{name}_slope_{p}d_D")
                cols.add(f"{name}_accel_{p}d_D")

    # BaseAdvancedStructuralMetrics
    for name in BaseAdvancedStructuralMetrics.CORE_METRICS.keys():
        cols.add(f"{name}_D")
        if name not in BaseAdvancedStructuralMetrics.SLOPE_ACCEL_EXCLUSIONS:
            for p in BaseAdvancedStructuralMetrics.UNIFIED_PERIODS:
                cols.add(f"{name}_slope_{p}d_D")
                cols.add(f"{name}_accel_{p}d_D")

    # BasePlatformFeature
    # Note: high, low, vpoc, total_volume are also present in OHLCV, but here they refer to platform-specific values.
    for field in ['platform_conviction_score', 'quality_score', 'duration', 'high', 'low', 'vpoc', 'total_volume', 'breakout_readiness_score', 'goodness_of_fit_score', 'precise_vpoc', 'internal_accumulation_intensity', 'breakout_quality_score', 'platform_character', 'character_score', 'platform_archetype']:
        cols.add(f"{field}_D")
    
    # BaseTrendlineFeature
    for field in ['slope', 'intercept', 'touch_points', 'validity_score', 'touch_conviction_score']:
        cols.add(f"{field}_D")

    # BaseMultiTimeframeTrendline
    # Note: slope, intercept, validity_score are also present in BaseTrendlineFeature, but here they are for MTF.
    # We assume these are distinct enough or will be handled by specific naming.
    for field in ['slope', 'intercept', 'validity_score', 'trend_conviction_score', 'period', 'line_type']:
        cols.add(f"{field}_D")
    
    # FundFlow (Tushare moneyflow) - these are already suffixed _D by DAO
    # The DAO for fund_flow_tushare already adds _D suffix.
    # Example: net_mf_amount_D, buy_sm_amount_D etc.
    # These are implicitly authoritative as they are specific fund flow metrics.

    # FundFlowCntTHS / FundFlowCntDC - these are already suffixed _ths / _dc by DAO
    # These are distinct and won't conflict with _D columns, so no special handling needed here.

    return cols

# Pre-calculate authoritative columns once to avoid repeated computation
AUTHORITATIVE_DAILY_COLS = _get_authoritative_daily_cols()

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
        self.feature_service = FeatureEngineeringService(self.calculator) # 修改代码行：注入 self.calculator
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
        【V4.1 · 衍生指标基名_D后缀修复版】预计算衍生指标列名适配器
        - 核心修复: 确保从DAO获取的预计算衍生指标（如 `base_name_slope_Xd`）在重命名时，
                  其 `base_name` 部分能够正确地继承 `_D` 后缀，如果对应的基础列（如 `base_name_D`）存在。
                  解决了 `SLOPE_X_base_name` 缺少 `_D` 后缀的问题。
        """
        import re
        rename_map = {}
        # 模式解释:
        #   (.+?)                     - 非贪婪匹配基础指标名 (base_name_raw)
        #   (?:_sum_(\d+)d)?          - 可选的sum部分，捕获sum周期 (sum_period)
        #   _(slope|accel)_(\d+)d$    - 必须匹配的斜率/加速度部分，捕获类型(deriv_type)和周期(deriv_period)
        pattern = re.compile(r'(.+?)(?:_sum_(\d+)d)?_(slope|accel)_(\d+)d$')
        
        # 获取当前DataFrame中所有列名，用于检查是否存在_D后缀版本
        current_columns = set(df.columns)

        for col in df.columns:
            match = pattern.match(col)
            if match:
                base_name_raw, sum_period, deriv_type, deriv_period = match.groups()
                
                # 确定最终的 base_name。
                # 如果 base_name_raw 不以 '_D' 结尾，但其 '_D' 后缀版本存在于 DataFrame 中，
                # 则使用 '_D' 后缀版本作为目标 base_name。
                # 否则，使用原始的 base_name_raw。
                target_base_name = base_name_raw
                # 修改代码行：确保只有当原始基名不带_D后缀，且其_D后缀版本存在时才添加
                if not base_name_raw.endswith('_D') and f"{base_name_raw}_D" in current_columns:
                    target_base_name = f"{base_name_raw}_D"

                # 根据捕获组构建新的、标准化的列名
                if sum_period:
                    new_name = f"{deriv_type.upper()}_{deriv_period}_{target_base_name}_sum_{sum_period}d"
                else:
                    new_name = f"{deriv_type.upper()}_{deriv_period}_{target_base_name}"
                
                rename_map[col] = new_name
        
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
        【V8.4 · OCH时序修复版】为策略准备数据的统一入口。
        - 核心修复: 调整了 `calculate_och` 的执行时序，确保其在所有依赖的元特征和上下文信号（如波动率不稳定性、市场情绪）计算并合并到DataFrame之后才执行。
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
        # --- 8. 【高级模式识别】 ---
        # 注意：此步骤依赖于一些斜率和加速度，但如果这些斜率和加速度是基于基础OHLCV或早期计算的指标，
        # 且不依赖于后续的上下文注入信号，则可以保留在此处。
        # 如果高级模式识别也依赖于上下文注入信号的斜率/加速度，则需要将其移动到斜率/加速度计算之后。
        # 暂时保留在此处，假设其依赖的斜率和加速度已在_prepare_base_data_and_indicators中处理或不依赖上下文信号。
        all_dfs = await self.feature_service.calculate_pattern_recognition_signals(all_dfs, config)
        # 【新增代码行】9. 【几何形态特征计算】 ---
        # 确保几何形态特征在斜率和加速度计算之前可用
        all_dfs = await self.feature_service.calculate_geometric_features(all_dfs, config)
        # --- 10. 【上下文信息注入】 ---
        # 此处将所有外部信号（包括smart_money_signals_df）合并到all_dfs['D']
        if not all_dfs or 'D' not in all_dfs or all_dfs['D'].empty:
            return all_dfs
        df_daily = all_dfs['D']
        start_date = df_daily.index.min().date()
        end_date = df_daily.index.max().date()
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
                    if pd.api.types.is_numeric_dtype(smart_money_signals_df[col]):
                        df_daily[col] = df_daily[col].fillna(0.0).astype(float)
                    else:
                        df_daily[col] = df_daily[col].fillna(False).astype(bool)
        all_dfs['D'] = df_daily # 更新all_dfs['D']为包含所有上下文信号的df_daily

        # NEW STEP: Calculate OCH_D for the daily dataframe (移动到此处)
        if 'D' in all_dfs and not all_dfs['D'].empty:
            temp_dfs = {'D': all_dfs['D']} # 重新封装，确保传入的是字典
            temp_dfs = await self.feature_service.calculate_och(temp_dfs)
            all_dfs['D'] = temp_dfs['D']

        # --- 11. 【斜率与加速度计算】(移动到所有上下文信息注入之后) ---
        all_dfs = await self.feature_service.calculate_all_slopes(all_dfs, config)
        all_dfs = await self.feature_service.calculate_all_accelerations(all_dfs, config)
        self._log_final_data_columns(all_dfs) # 移除调试打印
        return all_dfs

    async def _prepare_base_data_and_indicators(
        self,
        stock_code: str,
        config: dict,
        trade_time: Optional[str] = None,
        latest_only: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        【V8.29 · 数据合并与命名统一重构版】
        - 核心重构: 彻底重构数据合并与命名规则，以 `_get_ohlcv_data` 为基础，建立清晰完整的日线数据 `df_daily_master`。
        - 统一命名: 所有从DAO获取的预计算数据，在合并前统一添加 `_D` 后缀，确保命名规范。
        - 健壮合并: 采用迭代合并策略，并明确定义哪些补充数据源的列具有“权威性”，优先保留其值。
        - 消除补丁: 移除复杂的冲突解决逻辑，通过统一的预处理和合并规则，减少后续打补丁的需求。
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
        
        # 确定需要获取的K线数量
        if latest_only:
            max_lookback = self._get_max_lookback_period(config)
            safety_buffer = 100
            base_needed_bars = max_lookback + safety_buffer
            print(f"    - [闪电模式启动] 策略最大回溯期: {max_lookback}, 安全缓冲: {safety_buffer}, 最终加载: {base_needed_bars} 条记录。")
        else:
            base_needed_bars = config.get('feature_engineering_params', {}).get('base_needed_bars', 1200)
        
        # 确定需要获取的基础时间框架（日线、分钟线）以及需要从日线重采样的目标时间框架
        base_tfs_to_fetch = set()
        resample_map = {}
        for tf in required_tfs:
            if tf in ['W', 'M']:
                base_tfs_to_fetch.add('D')
                resample_map[tf] = 'D'
            else:
                base_tfs_to_fetch.add(tf)
        
        indicators_config = config.get('feature_engineering_params', {}).get('indicators', {})
        
        # --- 1. 集中定义所有数据获取任务 ---
        # 每个任务是一个元组: (tag, fetch_coroutine, is_advanced_metric_model_tag)
        # is_advanced_metric_model_tag 用于指示是否需要应用 _rename_precomputed_derivatives
        data_fetch_tasks: List[Tuple[str, Awaitable[pd.DataFrame], bool]] = []

        # OHLCV 数据 (基础数据)
        for tf in base_tfs_to_fetch:
            data_fetch_tasks.append((tf, self._get_ohlcv_data(stock_code, tf, base_needed_bars, trade_time), False))

        # 补充数据 (预计算特征)
        trade_time_dt = pd.to_datetime(trade_time, utc=True) if trade_time else None
        trade_time_dt_date = trade_time_dt.date() if trade_time_dt else datetime.datetime.now().date()

        # legacy_supplemental (fund_flow_and_chips_data)
        data_fetch_tasks.append(('legacy_supplemental', self.strategies_dao.get_fund_flow_and_chips_data(stock_code, trade_time_dt, base_needed_bars), False))
        # advanced_chips
        data_fetch_tasks.append(('advanced_chips', self.strategies_dao.get_advanced_chip_metrics_data(stock_code, trade_time_dt, base_needed_bars), True))
        # daily_basic
        data_fetch_tasks.append(('daily_basic', self.strategies_dao.get_daily_basic_data(stock_code, trade_time_dt, base_needed_bars), False))
        # fund_flow_ths (already suffixes _ths in DAO)
        data_fetch_tasks.append(('fund_flow_ths', self.fund_flow_dao.get_fund_flow_ths_data(stock_code, trade_time_dt_date, base_needed_bars), False))
        # fund_flow_dc (already suffixes _dc in DAO)
        data_fetch_tasks.append(('fund_flow_dc', self.fund_flow_dao.get_fund_flow_dc_data(stock_code, trade_time_dt_date, base_needed_bars), False))
        # fund_flow_tushare (already suffixes _D in DAO)
        data_fetch_tasks.append(('fund_flow_tushare', self.fund_flow_dao.get_fund_flow_daily_data(stock_code, trade_time_dt_date, base_needed_bars), False))
        # advanced_fund_flow
        data_fetch_tasks.append(('advanced_fund_flow', self.fund_flow_dao.get_advanced_fund_flow_metrics_data(stock_code, trade_time_dt_date, base_needed_bars), True))
        # price_limit
        data_fetch_tasks.append(('price_limit', self.strategies_dao.get_price_limit_data(stock_code, trade_time_dt, base_needed_bars), False))
        # advanced_structural_metrics
        data_fetch_tasks.append(('advanced_structural_metrics', self.strategies_dao.get_advanced_structural_metrics_data(stock_code, trade_time_dt, base_needed_bars), True))
        # platform_feature
        data_fetch_tasks.append(('platform_feature', self.strategies_dao.get_platform_feature_data(stock_code, trade_time_dt, base_needed_bars), True))
        # trendline_feature
        data_fetch_tasks.append(('trendline_feature', self.strategies_dao.get_trendline_feature_data(stock_code, trade_time_dt, base_needed_bars), True))
        # multi_timeframe_trendline
        data_fetch_tasks.append(('multi_timeframe_trendline', self.strategies_dao.get_multi_timeframe_trendline_data(stock_code, trade_time_dt, base_needed_bars), True))

        # 执行所有数据获取任务
        results = await asyncio.gather(*[task[1] for task in data_fetch_tasks], return_exceptions=True)
        
        # --- 2. 初始化 df_daily_master 并处理所有数据 ---
        all_dfs: Dict[str, pd.DataFrame] = {}
        df_daily_master: Optional[pd.DataFrame] = None
        
        # 2.1. 首先处理 OHLCV 数据，建立 df_daily_master
        for i, (tag, _, _) in enumerate(data_fetch_tasks):
            if tag in base_tfs_to_fetch: # 仅处理 OHLCV 相关的 tag
                df_ohlcv = results[i]
                if df_ohlcv is None or isinstance(df_ohlcv, Exception) or df_ohlcv.empty:
                    if isinstance(df_ohlcv, Exception):
                        logger.error(f"获取 {tag} OHLCV 数据失败: {df_ohlcv}", exc_info=False)
                    else:
                        logger.warning(f"[{stock_code}] 时间级别 {tag} OHLCV 数据为空。")
                    continue
                
                df_ohlcv = self._standardize_df_index_to_utc(df_ohlcv)
                df_ohlcv.index = df_ohlcv.index.normalize()
                if df_ohlcv.index.duplicated().any():
                    df_ohlcv = df_ohlcv[~df_ohlcv.index.duplicated(keep='last')]
                
                # 统一 OHLCV 列名并添加 _D 后缀
                ohlcv_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
                rename_ohlcv_map = {col: f"{col}_D" for col in ohlcv_cols if col in df_ohlcv.columns and not col.endswith('_D')}
                df_ohlcv.rename(columns=rename_ohlcv_map, inplace=True)
                
                if tag == 'D':
                    df_daily_master = df_ohlcv
                else:
                    all_dfs[tag] = df_ohlcv # 存储分钟线等非日线OHLCV
        
        if df_daily_master is None or df_daily_master.empty:
            print(f"    - 错误: 最核心的日线数据获取失败，处理终止。")
            return {}

        # 2.2. 迭代处理并合并所有补充数据到 df_daily_master
        # 记录所有被添加或更新的列，用于后续的ffill
        cols_to_ffill_after_merge = set()

        for i, (tag, _, is_advanced_metric_model) in enumerate(data_fetch_tasks):
            if tag in base_tfs_to_fetch: # OHLCV数据已处理，跳过
                continue

            df_supp = results[i]
            if df_supp is None or isinstance(df_supp, Exception) or df_supp.empty:
                if isinstance(df_supp, Exception):
                    logger.error(f"获取 {tag} 数据失败: {df_supp}", exc_info=False)
                else:
                    logger.warning(f"获取 {tag} 数据为空或无效。")
                continue
            
            df_supp = self._standardize_df_index_to_utc(df_supp)
            df_supp.index = df_supp.index.normalize()
            
            # 统一列名：为所有非时间框架后缀的列添加 _D 后缀
            rename_map_suffix = {}
            for col in df_supp.columns:
                # 避免重复添加 _D，或修改已有的 _W/_M/_ths/_dc 后缀
                if not col.endswith(('_D', '_W', '_M', '_ths', '_dc')):
                    rename_map_suffix[col] = f"{col}_D"
            df_supp.rename(columns=rename_map_suffix, inplace=True)

            # 应用 _rename_precomputed_derivatives (针对高级指标模型)
            if is_advanced_metric_model:
                df_supp = self._rename_precomputed_derivatives(df_supp)
            
            # 记录 df_supp 中将要合并的列
            current_supp_cols = set(df_supp.columns)
            cols_to_ffill_after_merge.update(current_supp_cols)

            # 合并策略：使用 df_daily_master.update() 优先保留补充数据的值
            # update() 方法会用 df_supp 中非NaN的值更新 df_daily_master 中对应列的值
            # 对于 df_supp 中有而 df_daily_master 中没有的列，update() 会添加它们
            # 对于 df_supp 中没有而 df_daily_master 中有的列，df_daily_master 的值不变
            # 这种方式天然地实现了“补充数据优先”的策略，因为补充数据通常是更精细的预计算特征。
            df_daily_master.update(df_supp)
            
            # 额外处理：对于那些在 AUTHORITATIVE_DAILY_COLS 中，但 df_supp 中可能为 NaN 的列，
            # 确保 df_daily_master 中的原始值不会覆盖 df_supp 中的 NaN。
            # update() 默认不会用 NaN 覆盖非 NaN。
            # 如果 df_supp 某个权威列是 NaN，而 df_daily_master 对应列有值，
            # 那么 df_daily_master 的值会保留。这通常是期望的行为。
            # 如果需要强制覆盖为 NaN，则需要更复杂的 merge + fillna 逻辑。
            # 目前的 update() 行为是合理的。

        # 2.3. 计算 pct_change_D
        if 'close_D' in df_daily_master.columns:
            df_daily_master['pct_change_D'] = df_daily_master['close_D'].pct_change().fillna(0)
        
        # 2.4. 计算 AAA_D 和 NMFNF_D (这些依赖于基础OHLCV和daily_basic中的total_market_value_D)
        # 确保这些计算在所有基础数据合并完成后进行
        temp_dfs_for_calc = {'D': df_daily_master}
        temp_dfs_for_calc = await self.feature_service.calculate_aaa_indicator(temp_dfs_for_calc)
        temp_dfs_for_calc = await self.feature_service.calculate_nmfnf(temp_dfs_for_calc)
        df_daily_master = temp_dfs_for_calc['D']

        # 2.5. 对所有新添加或更新的列进行 ffill
        # 确保只对实际存在于 df_daily_master 中的列进行 ffill
        final_cols_to_ffill = [col for col in cols_to_ffill_after_merge if col in df_daily_master.columns]
        if final_cols_to_ffill:
            df_daily_master[final_cols_to_ffill] = df_daily_master[final_cols_to_ffill].ffill()
            logger.info(f"已对 {len(final_cols_to_ffill)} 个补充数据列进行前向填充。")

        all_dfs['D'] = df_daily_master

        # 2.6. 重采样周线/月线数据并计算其合成指标
        if resample_map:
            for target_tf, source_tf in resample_map.items():
                if source_tf == 'D' and not df_daily_master.empty:
                    aggregation_rules = {
                        'open_D': 'first', 'high_D': 'max', 'low_D': 'min', 'close_D': 'last', 'volume_D': 'sum'
                    }
                    # 动态为所有已有的 _D 后缀列添加聚合规则
                    for col in df_daily_master.columns:
                        if col.endswith('_D') and col not in aggregation_rules:
                            # 默认对数值型指标取最后值，对金额/成交量/计数类指标求和
                            if 'amount' in col.lower() or 'vol' in col.lower() or 'net' in col.lower() or 'flow' in col.lower() or 'value' in col.lower() or 'num' in col.lower() or 'count' in col.lower():
                                aggregation_rules[col] = 'sum'
                            # 对于斜率和加速度，通常取最后值
                            elif '_slope_' in col.lower() or '_accel_' in col.lower():
                                aggregation_rules[col] = 'last'
                            # 对于其他指标，如率、比、分数、指数、熵、信念、稳固度、张力、偏度、Alpha、效率、强度、纯度、质量、动量、惯性、稳定性、能量、乖离、CMF、RSI、MACD、DMI、ROC、BOLL、ATR、OBV、KDJ、UO、VWAP、DMA、ATAN、ZIGZAG、OCH、NMFNF 等，取最后值
                            elif 'rate' in col.lower() or 'ratio' in col.lower() or 'pct' in col.lower() or 'score' in col.lower() or 'index' in col.lower() or 'entropy' in col.lower() or 'conviction' in col.lower() or 'solidity' in col.lower() or 'tension' in col.lower() or 'skew' in col.lower() or 'alpha' in col.lower() or 'efficiency' in col.lower() or 'strength' in col.lower() or 'intensity' in col.lower() or 'purity' in col.lower() or 'quality' in col.lower() or 'momentum' in col.lower() or 'inertia' in col.lower() or 'stability' in col.lower() or 'energy' in col.lower() or 'bias' in col.lower() or 'cmf' in col.lower() or 'rsi' in col.lower() or 'macd' in col.lower() or 'dmi' in col.lower() or 'roc' in col.lower() or 'boll' in col.lower() or 'atr' in col.lower() or 'obv' in col.lower() or 'kdj' in col.lower() or 'uo' in col.lower() or 'vwap' in col.lower() or 'dma' in col.lower() or 'atan' in col.lower() or 'zigzag' in col.lower() or 'och' in col.lower() or 'nmfnf' in col.lower():
                                aggregation_rules[col] = 'last'
                            else:
                                aggregation_rules[col] = 'last' # 默认取最后值
                    
                    # 特殊处理：换手率通常取平均
                    if 'turnover_rate_f_D' in aggregation_rules:
                        aggregation_rules['turnover_rate_f_D'] = 'mean'
                    if 'turnover_rate_D' in aggregation_rules:
                        aggregation_rules['turnover_rate_D'] = 'mean'

                    resample_period = 'W-FRI' if target_tf == 'W' else 'ME'
                    df_resampled = df_daily_master.resample(resample_period).agg(aggregation_rules)
                    df_resampled.dropna(how='all', inplace=True)
                    
                    if not df_resampled.empty:
                        # 重命名列，将 _D 替换为目标时间框架后缀
                        rename_map_resampled = {col: col.replace('_D', f'_{target_tf}') for col in df_resampled.columns if col.endswith('_D')}
                        df_resampled.rename(columns=rename_map_resampled, inplace=True)
                        
                        # 计算合成周线指标 (如 CMF, RSI)
                        if target_tf == 'W':
                            df_synthetic_indicators = self._calculate_synthetic_weekly_indicators(df_daily_master, df_resampled)
                            df_resampled = df_resampled.merge(df_synthetic_indicators, left_index=True, right_index=True, how='left')
                        
                        all_dfs[target_tf] = df_resampled
        
        processed_dfs: Dict[str, pd.DataFrame] = {}
        calc_tasks = []
        
        # 2.7. 为所有时间框架计算技术指标
        async def _calculate_for_tf(tf, df):
            df = self._standardize_df_index_to_utc(df)
            df_with_indicators = await self._calculate_indicators_for_timescale(df, indicators_config, tf)
            return tf, df_with_indicators
        
        for tf, df in all_dfs.items(): # 遍历所有已准备好的时间框架数据 (D, W, M, 60min etc.)
            if tf in required_tfs: # 确保只计算配置中需要的时间框架
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
        【V110.23 · 指标计算前重复索引终极清理版】根据配置为指定时间周期计算所有技术指标。
        - 核心修复: 在指标计算前，对 `df_for_calc` 进行最终的重复索引清理，确保 `pandas_ta` 等库接收到的DataFrame索引唯一。
        - 核心修复: 调整了 `merge_results` 函数的逻辑，确保只对**不以时间框架后缀结尾**的列添加后缀，避免重复。
        - 【修复】对于 `advanced_structural_metrics`, `platform_feature`, `trendline_feature`, `multi_timeframe_trendline`，
                  这些数据已在数据准备阶段从DAO获取并合并到DataFrame中，此处不再尝试调用 `IndicatorCalculator` 进行计算，而是直接跳过。
        """
        if not config:
            return df
        max_required_period = self._get_max_period_for_timeframe(config, timeframe_key)
        if len(df) < max_required_period:
            logger.warning(f"数据行数 ({len(df)}) 不足以满足周期 '{timeframe_key}' 的最大计算要求 ({max_required_period})，将跳过。")
            return df
        df_for_calc = df.copy()

        # 新增代码块：在指标计算前，再次检查并清理重复索引
        if df_for_calc.index.duplicated().any():
            initial_len = len(df_for_calc)
            df_for_calc = df_for_calc[~df_for_calc.index.duplicated(keep='last')]
            print(f"调试信息: 在 _calculate_indicators_for_timescale 中发现并清理了 {initial_len - len(df_for_calc)} 个重复索引 (周期: {timeframe_key})。")

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
            'fibonacci_levels', 'dma', 'atan_ma_angle', 'ma_velocity_acceleration', 'zigzag',
            # 【新增代码行】将结构与形态指标的计算放在最后，因为它们通常是直接从DAO获取并标准化后的数据，不需要复杂的依赖
            'advanced_structural_metrics', 'platform_feature', 'trendline_feature', 'multi_timeframe_trendline'
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
            # 【修改代码块】处理结构与形态指标，它们不需要 periods 循环，直接处理即可
            if indicator_name in ['advanced_structural_metrics', 'platform_feature', 'trendline_feature', 'multi_timeframe_trendline']:
                # 这些指标的数据已在 _prepare_base_data_and_indicators 中从DAO获取并合并到df_for_calc中，
                # 并且已经带有正确的后缀。此处无需再次计算或处理，直接跳过。
                logger.debug(f"指标 '{indicator_name}' (周期: {timeframe_key}) 已在数据准备阶段处理，跳过重复计算。")
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









