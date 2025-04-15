import asyncio
import warnings
import logging
import pandas as pd
import numpy as np
from django.utils import timezone
from typing import Any, List, Optional, Union, Dict, Tuple
import pandas_ta as ta

# --- 添加以下代码来忽略 FutureWarning ---
# 仅忽略 FutureWarning
# warnings.simplefilter(action='ignore', category=FutureWarning)
# ---------------------------------------
# --- 添加以下代码来忽略特定的 UserWarning ---
# 仅忽略关于 "drop timezone information" 的 UserWarning
warnings.filterwarnings(action='ignore', category=UserWarning, message='.*drop timezone information.*')
# -------------------------------------------

# 解决 pandas_ta 可能出现的 SettingWithCopyWarning，虽然通常不影响结果
pd.options.mode.chained_assignment = None # default='warn'

from dao_manager.daos.indicator_dao import IndicatorDAO
from dao_manager.daos.stock_basic_dao import StockBasicDAO
from core.constants import TimeLevel, FIB_PERIODS

logger = logging.getLogger("services")

class IndicatorService:
    """
    技术指标计算服务 (使用 pandas-ta)
    """
    def __init__(self):
        self.indicator_dao = IndicatorDAO()
        self.stock_basic_dao = StockBasicDAO() # 用于获取 StockInfo 对象
        # 动态导入 pandas_ta，避免在类实例化时就强制依赖
        try:
            global ta
            import pandas_ta as ta
        except ImportError:
            logger.error("pandas-ta 库未安装，请运行 'pip install pandas-ta'")
            ta = None # 设置为 None，后续计算会失败但不会在导入时崩溃

    async def _get_ohlcv_data(self, stock_code: str, time_level: Union[TimeLevel, str], needed_bars: int) -> Optional[pd.DataFrame]:
        """获取足够用于计算的历史数据"""
        limit = needed_bars # 增加一些 buffer
        # logger.debug(f"为计算指标 {stock_code} {time_level}，尝试获取 {limit} 条历史数据")
        df = await self.indicator_dao.get_history_ohlcv_df(stock_code, time_level, limit=limit)
        if df is None or df.empty:
            logger.warning(f"无法获取足够的历史数据来计算指标: {stock_code} {time_level}")
            return None
        # --- 添加代码：将 stock_code 和 time_level 列转换为 category 类型 (如果存在) ---
        if 'stock_code' in df.columns:
            try:
                df['stock_code'] = df['stock_code'].astype('category')
            except Exception as e:
                logger.warning(f"转换 stock_code 列为 category 类型失败: {e}")

        if 'time_level' in df.columns: # 假设你的 DataFrame 可能有 time_level 列
            try:
                df['time_level'] = df['time_level'].astype('category')
            except Exception as e:
                logger.warning(f"转换 time_level 列为 category 类型失败: {e}")
        # --------------------------------------------------------------------
        # if len(df) < needed_bars:
        #      logger.warning(f"获取到的历史数据 ({len(df)}) 少于所需 ({needed_bars}): {stock_code} {time_level}")
             # return None # 保持原有逻辑，即使数据不足也尝试计算
        # 确保列名是小写，pandas-ta 推荐使用小写列名
        df.columns = [col.lower() for col in df.columns]
        # 检查必需列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"获取的数据缺少必需列 (open, high, low, close, volume): {stock_code} {time_level}, 列: {df.columns.tolist()}")
            return None
        # 如果有 turnover 列，也转为小写
        if 'turnover' in df.columns:
            df = df.rename(columns={'turnover': 'turnover'}) # 确保是小写
        # logger.info(f"获取的数据长度: {len(df)}")
        
        return df

    # --- 单个指标计算方法 (使用 pandas-ta) ---

    def calculate_atr_fib(self, ohlc: pd.DataFrame) -> Optional[pd.DataFrame]:
        """计算不同斐波那契周期的 ATR"""
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty: return None
        try:
            results = {}
            for period in FIB_PERIODS:
                if len(ohlc) >= period:
                    # pandas-ta atr 返回 Series，名称为 ATR_period
                    atr_series = ohlc.ta.atr(length=period)
                    if atr_series is not None:
                        results[f'ATR_{period}'] = atr_series # 使用与之前一致的 key
                # else:
                #     logger.warning(f"数据不足 ({len(ohlc)}) 无法计算 ATR({period})")
            return pd.DataFrame(results, index=ohlc.index) if results else None
        except Exception as e:
            logger.error(f"计算 ATR_FIB 失败: {e}", exc_info=True)
            return None

    def calculate_boll(self, ohlc: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> Optional[pd.DataFrame]:
        """计算布林带 (标准周期)"""
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < period: return None
        try:
            # pandas-ta bbands 返回 DataFrame，包含多列，如 BBL_period_std, BBM_period_std, BBU_period_std 等
            bbands_df = ohlc.ta.bbands(length=period, std=std_dev)
            if bbands_df is None or bbands_df.empty:
                return None
            # 重命名列以匹配 finta 的输出 ('BB_UPPER', 'BB_MIDDLE', 'BB_LOWER')，以便与 DAO 兼容
            rename_map = {
                f'BBU_{period}_{std_dev}': 'BB_UPPER',
                f'BBM_{period}_{std_dev}': 'BB_MIDDLE',
                f'BBL_{period}_{std_dev}': 'BB_LOWER',
            }
            # 只保留并重命名所需的列
            required_cols = list(rename_map.keys())
            # 检查实际返回的列名，因为 pandas-ta 的命名可能微调
            actual_cols = {col: rename_map[col] for col in required_cols if col in bbands_df.columns}
            if len(actual_cols) != 3:
                 logger.warning(f"pandas-ta bbands 未返回所有预期列 for period={period}, std={std_dev}. 返回: {bbands_df.columns.tolist()}")
                 # 尝试使用可能存在的通用列名（如果 pandas-ta 版本变化）
                 potential_map = {'BBU': 'BB_UPPER', 'BBM': 'BB_MIDDLE', 'BBL': 'BB_LOWER'}
                 common_cols = {f'{k}_{period}_{std_dev}': v for k, v in potential_map.items() if f'{k}_{period}_{std_dev}' in bbands_df.columns}
                 if len(common_cols) == 3:
                     actual_cols = common_cols
                 else: # 仍然找不到，返回 None 或部分结果
                     logger.error(f"无法从 pandas-ta bbands 结果中提取所需列。")
                     # return None # 或者只返回能找到的列
                     found_df = bbands_df[[col for col in actual_cols.keys()]].rename(columns=actual_cols)
                     return found_df if not found_df.empty else None


            return bbands_df[list(actual_cols.keys())].rename(columns=actual_cols)

        except Exception as e:
            logger.error(f"计算 BOLL({period}, {std_dev}) 失败: {e}", exc_info=True)
            return None

    def calculate_cci_fib(self, ohlc: pd.DataFrame) -> Optional[pd.DataFrame]:
        """计算不同斐波那契周期的 CCI"""
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty: return None
        try:
            results = {}
            for period in FIB_PERIODS:
                if len(ohlc) >= period:
                    # pandas-ta cci 返回 Series，名称为 CCI_period
                    cci_series = ohlc.ta.cci(length=period)
                    if cci_series is not None:
                         # 使用与之前 finta 不同的 key，但更符合 pandas-ta 风格
                         # results[f'{period} period CCI'] = cci_series
                         results[f'CCI_{period}'] = cci_series # 改为 CCI_period 格式
                # else:
                #      logger.warning(f"数据不足 ({len(ohlc)}) 无法计算 CCI({period})")
            return pd.DataFrame(results, index=ohlc.index) if results else None
        except Exception as e:
            logger.error(f"计算 CCI_FIB 失败: {e}", exc_info=True)
            return None

    def calculate_cmf_fib(self, ohlc: pd.DataFrame) -> Optional[pd.DataFrame]:
        """计算不同斐波那契周期的 CMF"""
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or 'volume' not in ohlc.columns:
            logger.error("计算 CMF 需要 'volume' 列")
            return None
        try:
            results = {}
            for period in FIB_PERIODS:
                if len(ohlc) >= period:
                    # pandas-ta cmf 直接计算
                    cmf_series = ohlc.ta.cmf(length=period)
                    if cmf_series is not None:
                        # results[f'{period} period CMF'] = cmf_series # finta 风格 key
                        results[f'CMF_{period}'] = cmf_series # pandas-ta 风格 key
                # else:
                #      logger.warning(f"数据不足 ({len(ohlc)}) 无法计算 CMF({period})")
            return pd.DataFrame(results, index=ohlc.index) if results else None
        except Exception as e:
            logger.error(f"计算 CMF_FIB 失败: {e}", exc_info=True)
            return None

    def calculate_dmi_fib(self, ohlc: pd.DataFrame) -> Optional[pd.DataFrame]:
        """计算不同斐波那契周期的 DMI (+DI, -DI, ADX, ADXR)"""
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty: return None
        try:
            results = {}
            # DMI/ADX/ADXR 通常需要 period*2 或更多数据才能稳定
            min_data_multiplier = 2
            dmi_periods = [p for p in FIB_PERIODS if p >= 5] # DMI 周期通常不小于 5

            for period in dmi_periods:
                # 检查数据长度
                # ADXR 需要 period + period - 1 = 2*period - 1 的数据来计算第一个值
                # ADX 本身也需要类似长度，给足缓冲
                # 手动计算 ADXR 需要 ADX series.shift(period)，所以至少需要 adx计算所需长度 + period
                # adx 需要大约 2*period，所以总共约 3*period
                # 保持之前的 required_len，shift 会自动产生 NaN
                required_len = period * min_data_multiplier + 1

                if len(ohlc) >= required_len:
                    adx_series = None # 用于计算 ADXR
                    adxr_series = None # 存储计算结果
                    try:
                        # 1. 计算 ADX, +DI, -DI (pandas-ta adx 返回 DataFrame)
                        adx_df = ohlc.ta.adx(length=period)

                        # 2. 提取 ADX, +DI, -DI 并计算 ADXR
                        if adx_df is not None and not adx_df.empty:
                            adx_col = f'ADX_{period}'
                            dmp_col = f'DMP_{period}' # +DI
                            dmn_col = f'DMN_{period}' # -DI

                            # 提取 ADX, +DI, -DI
                            if adx_col in adx_df.columns:
                                adx_series = adx_df[adx_col] # 获取 ADX Series
                                results[f'ADX_{period}'] = adx_series
                            if dmp_col in adx_df.columns: results[f'+DI_{period}'] = adx_df[dmp_col]
                            if dmn_col in adx_df.columns: results[f'-DI_{period}'] = adx_df[dmn_col]

                            # 3. 手动计算 ADXR (如果成功获取了 ADX Series)
                            if adx_series is not None and isinstance(adx_series, pd.Series):
                                # ADXR = (ADX + ADX.shift(period)) / 2
                                # 确保有足够非 NaN 值进行 shift 操作
                                if adx_series.notna().sum() > period:
                                    adxr_series = (adx_series + adx_series.shift(period)) / 2
                                    # 将计算得到的 ADXR Series 添加到 results 中
                                    results[f'ADXR_{period}'] = adxr_series
                                # else:
                                #     logger.warning(f"ADX series for period {period} 不足以计算 ADXR (有效值数量 <= period)")
                        #     else:
                        #         logger.warning(f"未能从 adx_df 中提取到 ADX_{period} 列，无法计算 ADXR({period})")

                        # else:
                        #     logger.warning(f"pandas-ta adx({period}) 计算失败或返回空")

                    except Exception as e_dmi_calc:
                         # 捕获特定周期的计算错误，记录并继续下一个周期
                         logger.error(f"内部计算 DMI/ADX/ADXR({period}) 时出错: {e_dmi_calc}", exc_info=True)

                # else:
                #     logger.warning(f"数据不足 ({len(ohlc)} < {required_len}) 无法计算 DMI/ADX/ADXR({period})")

            # 过滤掉值为 None 或完全是 NaN 的 Series
            final_results = {k: v for k, v in results.items() if isinstance(v, pd.Series) and not v.isnull().all()}

            return pd.DataFrame(final_results, index=ohlc.index) if final_results else None
        except Exception as e:
            # 捕获整个函数级别的错误
            logger.error(f"计算 DMI_FIB 失败: {e}", exc_info=True)
            return None

    def calculate_ichimoku(self, ohlc: pd.DataFrame) -> Optional[pd.DataFrame]:
        """计算一目均衡表"""
        if ta is None: logger.error("pandas-ta 未加载"); return None
        # Ichimoku 默认参数 9, 26, 52。需要数据量 > 52
        min_required = 52 # Kijun=26, Senkou=52 lookbacks
        # if ohlc is None or ohlc.empty or len(ohlc) < min_required:
            # logger.warning(f"数据不足 ({len(ohlc)} < {min_required}) 可能无法准确计算 Ichimoku")
            # return None # 保持原有逻辑，尝试计算，pandas-ta 内部会处理

        try:
            # pandas-ta ichimoku 返回 DataFrame 和 SPAN A/B 的 shifted 版本
            # 我们通常需要未 shift 的版本: ITS_9, IKS_26, ICS_26, ISA_9, ISB_26 (根据日志调整)
            # 参数: tenkan=9, kijun=26, senkou=52
            ichi_df, span_shifted_df = ohlc.ta.ichimoku(tenkan=9, kijun=26, senkou=52)

            if ichi_df is None or ichi_df.empty:
                logger.warning("pandas-ta ichimoku 计算返回空")
                return None

            # 重命名列以匹配 finta 的输出 ('TENKAN', 'KIJUN', 'CHIKOU', 'SENKOU A', 'SENKOU B')
            # 注意：根据错误日志调整 ISA 和 ISB 的键名
            rename_map = {
                'ITS_9': 'TENKAN',         # Tenkan-sen (Conversion Line)
                'IKS_26': 'KIJUN',         # Kijun-sen (Base Line)
                'ICS_26': 'CHIKOU',        # Chikou Span (Lagging Span)
                # 'ISA_9_26_52': 'SENKOU A', # --- 旧的错误的键名 ---
                'ISA_9': 'SENKOU A',       # --- 根据日志调整后的键名 --- Senkou Span A (Leading Span A)
                # 'ISB_26_52': 'SENKOU B', # --- 旧的错误的键名 ---
                'ISB_26': 'SENKOU B',      # --- 根据日志调整后的键名 --- Senkou Span B (Leading Span B)
            }
            required_keys = list(rename_map.keys())
            actual_cols_map = {key: rename_map[key] for key in required_keys if key in ichi_df.columns}

            if len(actual_cols_map) != 5:
                # 记录更详细的错误，包括期望的和实际的
                expected_cols = required_keys
                returned_cols = ichi_df.columns.tolist()
                logger.error(f"无法从 pandas-ta ichimoku 结果中提取所有所需列。期望键名: {expected_cols}, 实际返回列: {returned_cols}")
                # 仍然尝试返回能找到的列，避免完全失败
                found_df = ichi_df[[key for key in actual_cols_map.keys()]].rename(columns=actual_cols_map)
                return found_df if not found_df.empty else None

            # 提取并重命名所需的列
            result_df = ichi_df[list(actual_cols_map.keys())].rename(columns=actual_cols_map)
            return result_df

        except Exception as e:
            logger.error(f"计算 Ichimoku 失败: {e}", exc_info=True)
            return None

    def calculate_kdj_fib(self, ohlc: pd.DataFrame, fast_k_period: int = 3, slow_k_period: int = 3) -> Optional[pd.DataFrame]:
        """计算不同斐波那契周期的 KDJ (N, M1, M2)"""
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty: return None
        try:
            results = {}
            m1 = fast_k_period
            m2 = slow_k_period
            for period in FIB_PERIODS: # period 对应 KDJ 中的 N
                if len(ohlc) >= period + m1 + m2:
                    try:
                        # pandas-ta kdj 返回 DataFrame
                        kdj_df = ohlc.ta.kdj(length=period, signal=m1, smooth_k=m2)

                        if kdj_df is not None and not kdj_df.empty:
                            # 从日志可知列名格式为: K_period_3, D_period_3, J_period_3
                            k_col = f'K_{period}_{m1}'
                            d_col = f'D_{period}_{m1}'
                            j_col = f'J_{period}_{m1}'
                            # 检查列名是否存在
                            if k_col in kdj_df.columns:
                                results[f'K_{period}'] = kdj_df[k_col]
                            if d_col in kdj_df.columns:
                                results[f'D_{period}'] = kdj_df[d_col]
                            if j_col in kdj_df.columns:
                                results[f'J_{period}'] = kdj_df[j_col]
                            # 如果通过列名未找到，回退到位置索引
                            if f'K_{period}' not in results and len(kdj_df.columns) >= 1:
                                results[f'K_{period}'] = kdj_df.iloc[:, 0]
                            if f'D_{period}' not in results and len(kdj_df.columns) >= 2:
                                results[f'D_{period}'] = kdj_df.iloc[:, 1]
                            if f'J_{period}' not in results and len(kdj_df.columns) >= 3:
                                results[f'J_{period}'] = kdj_df.iloc[:, 2]
                        # else:
                        #     logger.warning(f"pandas-ta kdj({period}, {m1}, {m2}) 计算失败或返回空")
                    except Exception as e_kdj_calc:
                        logger.error(f"内部计算 KDJ({period}, {m1}, {m2}) 时出错: {e_kdj_calc}", exc_info=True)
                # else:
                #     logger.warning(f"数据不足 ({len(ohlc)}) 无法计算 KDJ({period}, {m1}, {m2})")
            final_results = {k: v for k, v in results.items() if v is not None and not v.isnull().all()}
            return pd.DataFrame(final_results, index=ohlc.index) if final_results else None
        except Exception as e:
            logger.error(f"计算 KDJ_FIB 失败: {e}", exc_info=True)
            return None

    def calculate_ema_fib(self, ohlc: pd.DataFrame) -> Optional[pd.DataFrame]:
        """计算不同斐波那契周期的 EMA"""
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty: return None
        try:
            results = {}
            for period in FIB_PERIODS:
                if len(ohlc) >= period:
                    # pandas-ta ema 返回 Series，名称为 EMA_period
                    ema_series = ohlc.ta.ema(length=period)
                    if ema_series is not None:
                        results[f'EMA_{period}'] = ema_series # key 与 pandas-ta 列名一致
                # else:
                #     logger.warning(f"数据不足 ({len(ohlc)}) 无法计算 EMA({period})")
            return pd.DataFrame(results, index=ohlc.index) if results else None
        except Exception as e:
            logger.error(f"计算 EMA_FIB 失败: {e}", exc_info=True)
            return None

    def calculate_amount_ma_fib(self, ohlc: pd.DataFrame) -> Optional[pd.DataFrame]:
        """计算不同斐波那契周期的成交额 SMA"""
        if ta is None: logger.error("pandas-ta 未加载"); return None
        # 需要 'turnover' 列，并且已在 _get_ohlcv_data 中转为小写
        if ohlc is None or ohlc.empty or 'turnover' not in ohlc.columns:
            logger.error("计算 Amount MA 需要 'turnover' 列")
            return None
        try:
            results = {}
            for period in FIB_PERIODS:
                if len(ohlc) >= period:
                    # 使用 pandas-ta sma 计算 turnover 的 SMA
                    # 需要指定 close='turnover'
                    amt_ma_series = ohlc.ta.sma(close='turnover', length=period)
                    # pandas-ta 返回的列名可能是 SMA_period，我们需要重命名 key
                    if amt_ma_series is not None:
                        results[f'AMT_MA_{period}'] = amt_ma_series
                # else:
                #     logger.warning(f"数据不足 ({len(ohlc)}) 无法计算 Amount MA({period})")
            return pd.DataFrame(results, index=ohlc.index) if results else None
        except Exception as e:
            logger.error(f"计算 Amount MA FIB 失败: {e}", exc_info=True)
            return None

    def calculate_macd_fib(self, ohlc: pd.DataFrame, period_fast: int = 12, period_slow: int = 26, signal: int = 9) -> Optional[pd.DataFrame]:
        """计算标准 MACD(12, 26, 9) 和斐波那契周期的 EMA"""
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < period_slow + signal: return None
        try:
            # 1. 计算标准 MACD
            # pandas-ta macd 返回 DataFrame，列名 MACD_fast_slow_signal, MACDh_fast_slow_signal, MACDs_fast_slow_signal
            macd_df = ohlc.ta.macd(fast=period_fast, slow=period_slow, signal=signal)

            if macd_df is None or macd_df.empty:
                logger.warning(f"pandas-ta macd({period_fast}, {period_slow}, {signal}) 计算失败或返回空")
                macd_result_df = pd.DataFrame(index=ohlc.index) # 创建空 DataFrame 以便合并
            else:
                # 重命名列以匹配 finta 的输出 ('MACD', 'SIGNAL') 和添加 'MACD_HIST'
                macd_col = f'MACD_{period_fast}_{period_slow}_{signal}'
                signal_col = f'MACDs_{period_fast}_{period_slow}_{signal}'
                hist_col = f'MACDh_{period_fast}_{period_slow}_{signal}'

                rename_map = {}
                if macd_col in macd_df.columns: rename_map[macd_col] = 'MACD'     # DIFF
                if signal_col in macd_df.columns: rename_map[signal_col] = 'SIGNAL' # DEA
                if hist_col in macd_df.columns: rename_map[hist_col] = 'MACD_HIST' # Histogram

                if len(rename_map) < 2: # 至少要有 MACD 和 SIGNAL
                     logger.error(f"无法从 pandas-ta macd 结果中提取 MACD 或 SIGNAL 列。返回: {macd_df.columns.tolist()}")
                     macd_result_df = pd.DataFrame(index=ohlc.index)
                else:
                    macd_result_df = macd_df[list(rename_map.keys())].rename(columns=rename_map)

            # 2. 计算斐波那契 EMA (调用已修改的 EMA 函数)
            ema_fib_df = self.calculate_ema_fib(ohlc)

            # 3. 合并结果
            if ema_fib_df is not None and not ema_fib_df.empty:
                # 使用 outer join 以保留所有日期索引，填充 NaN
                result_df = pd.concat([macd_result_df, ema_fib_df], axis=1, join='outer')
            else:
                result_df = macd_result_df

            return result_df if not result_df.empty else None

        except Exception as e:
            logger.error(f"计算 MACD_FIB 失败: {e}", exc_info=True)
            return None

    def calculate_mfi_fib(self, ohlc: pd.DataFrame) -> Optional[pd.DataFrame]:
        """计算不同斐波那契周期的 MFI"""
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty: return None
        try:
            results = {}
            for period in FIB_PERIODS:
                if len(ohlc) >= period:
                    # --- 使用上下文管理器忽略警告 ---
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        # pandas-ta mfi 返回 Series，名称为 MFI_period
                        mfi_series = ohlc.ta.mfi(length=period)
                    # ---------------------------------
                    if mfi_series is not None:
                        results[f'MFI_{period}'] = mfi_series # key 与 pandas-ta 列名一致
                # else:
                #     logger.warning(f"数据不足 ({len(ohlc)}) 无法计算 MFI({period})")
            return pd.DataFrame(results, index=ohlc.index) if results else None
        except Exception as e:
            logger.error(f"计算 MFI_FIB 失败: {e}", exc_info=True)
            return None

    def calculate_mom_fib(self, ohlc: pd.DataFrame) -> Optional[pd.DataFrame]:
        """计算不同斐波那契周期的 MOM"""
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty: return None
        try:
            results = {}
            for period in FIB_PERIODS:
                if len(ohlc) >= period:
                    # pandas-ta mom 返回 Series，名称为 MOM_period
                    mom_series = ohlc.ta.mom(length=period)
                    if mom_series is not None:
                        results[f'MOM_{period}'] = mom_series # key 与 pandas-ta 列名一致
                # else:
                #     logger.warning(f"数据不足 ({len(ohlc)}) 无法计算 MOM({period})")
            return pd.DataFrame(results, index=ohlc.index) if results else None
        except Exception as e:
            logger.error(f"计算 MOM_FIB 失败: {e}", exc_info=True)
            return None

    def calculate_obv(self, ohlc: pd.DataFrame) -> Optional[pd.DataFrame]:
        """计算 OBV"""
        if ta is None: logger.error("pandas-ta 未加载"); return None
        # 假设 DAO 层已处理重复索引问题
        if ohlc is None or ohlc.empty: return None
        try:
            # pandas-ta obv 返回 Series，名称为 OBV
            obv_series = ohlc.ta.obv()
            if obv_series is None: return None
            # 返回 DataFrame，列名为 OBV
            return obv_series.to_frame(name='OBV')
        except Exception as e:
            logger.error(f"计算 OBV 失败: {e}", exc_info=True)
            return None

    def calculate_roc_fib(self, ohlc: pd.DataFrame) -> Optional[pd.DataFrame]:
        """计算不同斐波那契周期的 ROC"""
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty: return None
        try:
            results = {}
            for period in FIB_PERIODS:
                if len(ohlc) >= period + 1: # ROC 需要 n+1 条数据
                    # pandas-ta roc 返回 Series，名称为 ROC_period
                    roc_series = ohlc.ta.roc(length=period)
                    if roc_series is not None:
                        results[f'ROC_{period}'] = roc_series # key 与 pandas-ta 列名一致
                # else:
                #     logger.warning(f"数据不足 ({len(ohlc)}) 无法计算 ROC({period})")
            return pd.DataFrame(results, index=ohlc.index) if results else None
        except Exception as e:
            logger.error(f"计算 ROC_FIB 失败: {e}", exc_info=True)
            return None

    def calculate_amount_roc_fib(self, ohlc: pd.DataFrame) -> Optional[pd.DataFrame]:
        """计算不同斐波那契周期的成交额 ROC"""
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or 'turnover' not in ohlc.columns:
             logger.error("计算 Amount ROC 需要 'turnover' 列")
             return None
        try:
            results = {}
            for period in FIB_PERIODS:
                if len(ohlc) >= period + 1:
                    # 使用 pandas-ta roc 计算 turnover 的 ROC
                    # 需要指定 close='turnover'
                    aroc_series = ohlc.ta.roc(close='turnover', length=period)
                    # pandas-ta 返回的列名可能是 ROC_period，我们需要重命名 key
                    if aroc_series is not None:
                        results[f'AROC_{period}'] = aroc_series
                # else:
                #     logger.warning(f"数据不足 ({len(ohlc)}) 无法计算 Amount ROC({period})")
            # 处理 inf (pandas-ta roc 内部可能已处理，但以防万一)
            df_results = pd.DataFrame(results, index=ohlc.index)
            df_results.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_results if not df_results.empty else None
        except Exception as e:
            logger.error(f"计算 Amount ROC FIB 失败: {e}", exc_info=True)
            return None

    def calculate_rsi_fib(self, ohlc: pd.DataFrame) -> Optional[pd.DataFrame]:
        """计算不同斐波那契周期的 RSI"""
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty: return None
        try:
            results = {}
            for period in FIB_PERIODS:
                if len(ohlc) >= period + 1: # RSI 内部计算需要 diff
                    # pandas-ta rsi 返回 Series，名称为 RSI_period
                    rsi_series = ohlc.ta.rsi(length=period)
                    if rsi_series is not None:
                        results[f'RSI_{period}'] = rsi_series # key 与 pandas-ta 列名一致
                # else:
                #     logger.warning(f"数据不足 ({len(ohlc)}) 无法计算 RSI({period})")
            return pd.DataFrame(results, index=ohlc.index) if results else None
        except Exception as e:
            logger.error(f"计算 RSI_FIB 失败: {e}", exc_info=True)
            return None

    def calculate_sar(self, ohlc: pd.DataFrame, af: float = 0.02, max_af: float = 0.2) -> Optional[pd.DataFrame]:
        """计算 SAR (使用 pandas-ta 的 psar)"""
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < 2: return None
        try:
            # 使用 pandas-ta 的 psar 函数，注意参数名 af0 和 af_max
            # psar 返回一个包含多列的 DataFrame
            psar_df = ohlc.ta.psar(af0=af, af_max=max_af) # 使用 psar 并传入对应参数

            if psar_df is None or psar_df.empty:
                logger.warning("pandas-ta psar 计算返回空")
                return None

            # psar 返回的列名通常包含参数，例如 PSARl_0.02_0.2, PSARs_0.02_0.2
            # 我们需要找到实际的列名
            psarl_col = next((col for col in psar_df.columns if col.startswith('PSARl')), None)
            psars_col = next((col for col in psar_df.columns if col.startswith('PSARs')), None)

            if psarl_col and psars_col:
                # 合并 PSARl 和 PSARs 列得到单一的 SAR 值
                # 当 PSARl 为 NaN 时取 PSARs 的值，反之亦然
                sar_series = psar_df[psarl_col].fillna(psar_df[psars_col])
                # 返回 DataFrame，列名为 SAR，以兼容 DAO
                return sar_series.to_frame(name='SAR')
            else:
                logger.error(f"无法从 pandas-ta psar 结果中找到 PSARl 或 PSARs 列。返回列: {psar_df.columns.tolist()}")
                return None

        except Exception as e:
            # 捕获可能的其他错误，例如参数错误
            logger.error(f"计算 SAR (psar) 失败: {e}", exc_info=True)
            return None

    def calculate_vroc_fib(self, ohlc: pd.DataFrame) -> Optional[pd.DataFrame]:
        """计算不同斐波那契周期的成交量 ROC (VROC)"""
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or 'volume' not in ohlc.columns:
             logger.error("计算 VROC 需要 'volume' 列")
             return None
        try:
            results = {}
            for period in FIB_PERIODS:
                if len(ohlc) >= period + 1:
                    # 使用 pandas-ta roc 计算 volume 的 ROC
                    # 需要指定 close='volume'
                    vroc_series = ohlc.ta.roc(close='volume', length=period)
                    # pandas-ta 返回的列名可能是 ROC_period，我们需要重命名 key
                    if vroc_series is not None:
                        results[f'VROC_{period}'] = vroc_series
                # else:
                #     logger.warning(f"数据不足 ({len(ohlc)}) 无法计算 VROC({period})")
            # 处理 inf
            df_results = pd.DataFrame(results, index=ohlc.index)
            df_results.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_results if not df_results.empty else None
        except Exception as e:
            logger.error(f"计算 VROC FIB 失败: {e}", exc_info=True)
            return None

    def calculate_vwap(self, ohlc: pd.DataFrame) -> Optional[pd.DataFrame]:
        """计算 VWAP (日内 VWAP 需要 timestamp index)"""
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty: return None
        # pandas-ta vwap 依赖于 DatetimeIndex 来确定重置周期 (通常是每日)
        if not isinstance(ohlc.index, pd.DatetimeIndex):
            logger.warning("计算 VWAP 需要 DataFrame 的 index 是 DatetimeIndex。尝试转换...")
            try:
                original_index = ohlc.index
                ohlc.index = pd.to_datetime(ohlc.index)
            except Exception as e:
                logger.error(f"无法将 index 转换为 DatetimeIndex，VWAP 计算可能不准确或失败: {e}")
                # 恢复原始索引，避免影响其他计算
                ohlc.index = original_index
                # return None # 或者尝试计算，结果可能不对

        try:
            # pandas-ta vwap 返回 Series，名称 VWAP_D (假设是日线或更高频率)
            vwap_series = ohlc.ta.vwap()
            if vwap_series is None: return None
            # 返回 DataFrame，列名为 VWAP
            return vwap_series.to_frame(name='VWAP')
        except Exception as e:
            logger.error(f"计算 VWAP 失败: {e}", exc_info=True)
            return None
        finally:
            # 如果之前转换了索引，尝试恢复 (如果需要的话，但通常计算完即可)
            # if 'original_index' in locals():
            #     ohlc.index = original_index
            pass

    def calculate_wr_fib(self, ohlc: pd.DataFrame) -> Optional[pd.DataFrame]:
        """计算不同斐波那契周期的 Williams %R (WR)"""
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty: return None
        try:
            results = {}
            for period in FIB_PERIODS:
                if len(ohlc) >= period:
                    # pandas-ta Williams %R 函数名为 willr，返回 Series，名称 WILLR_period
                    wr_series = ohlc.ta.willr(length=period)
                    if wr_series is not None:
                        results[f'WR_{period}'] = wr_series # 重命名 key
                # else:
                #     logger.warning(f"数据不足 ({len(ohlc)}) 无法计算 WR({period})")
            return pd.DataFrame(results, index=ohlc.index) if results else None
        except Exception as e:
            logger.error(f"计算 WR_FIB 失败: {e}", exc_info=True)
            return None

    def calculate_sma_fib(self, ohlc: pd.DataFrame) -> Optional[pd.DataFrame]:
        """计算不同斐波那契周期的 SMA"""
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty: return None
        try:
            results = {}
            for period in FIB_PERIODS:
                if len(ohlc) >= period:
                    # pandas-ta sma 返回 Series，名称为 SMA_period
                    sma_series = ohlc.ta.sma(length=period)
                    if sma_series is not None:
                        results[f'SMA_{period}'] = sma_series # key 与 pandas-ta 列名一致
                # else:
                #     logger.warning(f"数据不足 ({len(ohlc)}) 无法计算 SMA({period})")
            return pd.DataFrame(results, index=ohlc.index) if results else None
        except Exception as e:
            logger.error(f"计算 SMA_FIB 失败: {e}", exc_info=True)
            return None

    def calculate_kc_fib(self, ohlc: pd.DataFrame, atr_length: int = 10, scalar: float = 2.0) -> Optional[pd.DataFrame]:
        """
        计算不同斐波那契周期的 Keltner Channels (KC)。
        使用斐波那契周期作为 EMA 的长度 (length)。
        ATR 长度固定或可配置。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty: return None
        try:
            all_kc_results = {}
            # KC 需要 EMA(period) 和 ATR(atr_length)
            min_len_required = max(FIB_PERIODS) + atr_length # 粗略估计
            if len(ohlc) < min_len_required:
                logger.warning(f"数据长度 ({len(ohlc)}) 可能不足以计算所有周期的 KC (需要约 {min_len_required})")
                # 仍然尝试计算，pandas-ta 会处理
            for period in FIB_PERIODS:
                if len(ohlc) >= max(period, atr_length): # 确保当前周期计算所需数据足够
                    try:
                        # 计算KC指标
                        kc_df = ohlc.ta.kc(length=period, atr_length=atr_length, scalar=scalar)
                        if kc_df is not None and not kc_df.empty:
                            # 根据日志发现的实际列名格式
                            lower_col = f'KCLe_{period}_{scalar}'
                            basis_col = f'KCBe_{period}_{scalar}'
                            upper_col = f'KCUe_{period}_{scalar}'
                            # 尝试直接匹配正确的列名
                            if lower_col in kc_df.columns:
                                all_kc_results[f'KC_LOWER_{period}'] = kc_df[lower_col]
                            if basis_col in kc_df.columns:
                                all_kc_results[f'KC_BASIS_{period}'] = kc_df[basis_col]
                            if upper_col in kc_df.columns:
                                all_kc_results[f'KC_UPPER_{period}'] = kc_df[upper_col]
                            # 如果没有找到匹配的列名，回退到位置索引
                            if f'KC_LOWER_{period}' not in all_kc_results and len(kc_df.columns) >= 1:
                                all_kc_results[f'KC_LOWER_{period}'] = kc_df.iloc[:, 0]
                            if f'KC_BASIS_{period}' not in all_kc_results and len(kc_df.columns) >= 2:
                                all_kc_results[f'KC_BASIS_{period}'] = kc_df.iloc[:, 1]
                            if f'KC_UPPER_{period}' not in all_kc_results and len(kc_df.columns) >= 3:
                                all_kc_results[f'KC_UPPER_{period}'] = kc_df.iloc[:, 2]
                                logger.info(f"使用位置索引(2)找到Upper值")
                        # else:
                        #     logger.warning(f"pandas-ta kc(length={period}, atr_length={atr_length}) 计算失败或返回空")
                    except Exception as e_kc_calc:
                        logger.error(f"内部计算 KC(length={period}, atr_length={atr_length}) 时出错: {e_kc_calc}", exc_info=True)
            # 过滤掉值为 None 或完全是 NaN 的 Series
            final_results = {k: v for k, v in all_kc_results.items() if isinstance(v, pd.Series) and not v.isnull().all()}
            return pd.DataFrame(final_results, index=ohlc.index) if final_results else None
        except Exception as e:
            logger.error(f"计算 KC_FIB 失败: {e}", exc_info=True)
            return None

    def calculate_stoch_fib(self, ohlc: pd.DataFrame, d_period: int = 3, smooth_k_period: int = 3) -> Optional[pd.DataFrame]:
        """
        计算不同斐波那契周期的 Stochastic Oscillator (%K, %D)。
        使用斐波那契周期作为 %K 的计算周期 (k)。
        %D 周期 (d) 和 Slow %K 周期 (smooth_k) 固定或可配置。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty: return None
        try:
            all_stoch_results = {}
            # Stochastic(k, d, smooth_k) 需要 k + d + smooth_k 左右数据
            min_len_required = max(FIB_PERIODS) + d_period + smooth_k_period

            if len(ohlc) < min_len_required:
                 logger.warning(f"数据长度 ({len(ohlc)}) 可能不足以计算所有周期的 Stochastic (需要约 {min_len_required})")

            for period in FIB_PERIODS: # period 作为 k
                if len(ohlc) >= period + d_period + smooth_k_period - 2: # 近似所需长度
                    try:
                        # pandas-ta stoch 返回 DataFrame，包含 STOCHk, STOCHd 列
                        # 列名格式: STOCHk_k_d_smooth, STOCHd_k_d_smooth
                        stoch_df = ohlc.ta.stoch(k=period, d=d_period, smooth_k=smooth_k_period)

                        if stoch_df is not None and not stoch_df.empty:
                            # 构建预期的列名
                            k_col = f'STOCHk_{period}_{d_period}_{smooth_k_period}'
                            d_col = f'STOCHd_{period}_{d_period}_{smooth_k_period}'

                            # 提取并重命名
                            if k_col in stoch_df.columns:
                                all_stoch_results[f'STOCH_K_{period}'] = stoch_df[k_col]
                            if d_col in stoch_df.columns:
                                all_stoch_results[f'STOCH_D_{period}'] = stoch_df[d_col]
                        else:
                            logger.warning(f"pandas-ta stoch(k={period}, d={d_period}, smooth_k={smooth_k_period}) 计算失败或返回空")
                    except Exception as e_stoch_calc:
                         logger.error(f"内部计算 Stochastic(k={period}, d={d_period}, smooth_k={smooth_k_period}) 时出错: {e_stoch_calc}", exc_info=True)
                # else:
                #     logger.warning(f"数据不足 ({len(ohlc)}) 无法计算 Stochastic(k={period}, d={d_period}, smooth_k={smooth_k_period})")

            final_results = {k: v for k, v in all_stoch_results.items() if isinstance(v, pd.Series) and not v.isnull().all()}
            return pd.DataFrame(final_results, index=ohlc.index) if final_results else None
        except Exception as e:
            logger.error(f"计算 STOCH_FIB 失败: {e}", exc_info=True)
            return None

    def calculate_adl(self, ohlc: pd.DataFrame) -> Optional[pd.DataFrame]:
        """计算 Accumulation/Distribution Line (ADL)"""
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or not all(c in ohlc.columns for c in ['high', 'low', 'close', 'volume']):
            logger.error("计算 ADL 需要 'high', 'low', 'close', 'volume' 列")
            return None
        try:
            # pandas-ta ADL 函数名为 ad
            adl_series = ohlc.ta.ad()
            if adl_series is None: return None
            # 返回 DataFrame，列名为 ADL
            return adl_series.to_frame(name='ADL')
        except Exception as e:
            logger.error(f"计算 ADL 失败: {e}", exc_info=True)
            return None

    def calculate_pivot_points(self, ohlc: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        计算 Pivot Points (PP, S1-S4, R1-R4)。
        依赖于 DatetimeIndex 来确定计算周期 (例如基于前一日计算当日)。
        """
        if ohlc is None or ohlc.empty: return None
        try:
            # 确保数据按日期排序
            ohlc = ohlc.sort_index()
            
            # 创建结果 DataFrame
            results = pd.DataFrame(index=ohlc.index)
            
            # 创建移位版本的价格数据来计算枢轴点
            # 使用前一交易日的 OHLC 计算当前交易日的枢轴点
            prev_high = ohlc['high'].shift(1)
            prev_low = ohlc['low'].shift(1)
            prev_close = ohlc['close'].shift(1)
            
            # 计算传统枢轴点
            # PP = (High + Low + Close) / 3
            PP = (prev_high + prev_low + prev_close) / 3
            results['PP'] = PP
            
            # 计算支撑位
            # S1 = (2 * PP) - High
            results['S1'] = (2 * PP) - prev_high
            # S2 = PP - (High - Low)
            results['S2'] = PP - (prev_high - prev_low)
            # S3 = S1 - (High - Low)
            results['S3'] = results['S1'] - (prev_high - prev_low)
            # S4 = S3 - (High - Low)
            results['S4'] = results['S3'] - (prev_high - prev_low)
            
            # 计算阻力位
            # R1 = (2 * PP) - Low
            results['R1'] = (2 * PP) - prev_low
            # R2 = PP + (High - Low)
            results['R2'] = PP + (prev_high - prev_low)
            # R3 = R1 + (High - Low)
            results['R3'] = results['R1'] + (prev_high - prev_low)
            # R4 = R3 + (High - Low)
            results['R4'] = results['R3'] + (prev_high - prev_low)
            
            # 可选：添加斐波那契枢轴点
            # 省略更多计算...
            
            # 删除第一行（无法计算）
            results = results.iloc[1:]
            
            return results
            
        except Exception as e:
            logger.error(f"计算 Pivot Points 失败: {e}", exc_info=True)
            return None

    # --- 统一计算和保存入口 (基本保持不变) ---
    async def calculate_and_save_all_indicators(self, stock_code: str, time_level: Union[TimeLevel, str]):
        """
        计算指定股票和时间级别的所有支持的指标，并保存到数据库。
        (已适配 pandas-ta 计算方法)
        """
        if ta is None:
            logger.error("pandas-ta 未加载，无法计算指标。请先安装 'pandas-ta'。")
            return
        # logger.info(f"开始计算和保存指标 for {stock_code} {time_level} using pandas-ta")
        stock_info = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock_info:
            logger.error(f"无法找到股票信息: {stock_code}，指标计算中止")
            return
        # 确定需要多少历史数据，取斐波那契最大值加上一些缓冲
        # 考虑 DMI/ADXR 等可能需要更多数据
        needed_bars = max(FIB_PERIODS) + 20 # 保持足够大的缓冲
        ohlcv_df_raw = await self._get_ohlcv_data(stock_code, time_level, needed_bars)
        if ohlcv_df_raw is None or ohlcv_df_raw.empty:
            logger.error(f"无法获取用于计算指标的历史数据 for {stock_code} {time_level}")
            return
        time_level_str = time_level.value if isinstance(time_level, TimeLevel) else str(time_level)
        # --- 逐个计算并保存 ---
        # 任务列表保持不变，因为函数签名和目的没变
        indicator_tasks = [
            (self.calculate_atr_fib, self.indicator_dao.save_atr_fib, {}),
            (self.calculate_boll, self.indicator_dao.save_boll, {}), # 使用默认参数 period=20
            (self.calculate_cci_fib, self.indicator_dao.save_cci_fib, {}),
            (self.calculate_cmf_fib, self.indicator_dao.save_cmf_fib, {}),
            (self.calculate_dmi_fib, self.indicator_dao.save_dmi_fib, {}),
            (self.calculate_ichimoku, self.indicator_dao.save_ichimoku, {}),
            (self.calculate_kdj_fib, self.indicator_dao.save_kdj_fib, {}), # 使用默认 M1=3, M2=3
            # EMA 和 MACD 一起计算，因为 MACD 模型包含 EMA
            (self.calculate_ema_fib, self.indicator_dao.save_ema_fib, {}), # 单独保存 EMA (可选)
            (self.calculate_amount_ma_fib, self.indicator_dao.save_amount_ma_fib, {}),
            (self.calculate_macd_fib, self.indicator_dao.save_macd_fib, {}), # 计算 MACD 和 EMA
            (self.calculate_mfi_fib, self.indicator_dao.save_mfi_fib, {}),
            (self.calculate_mom_fib, self.indicator_dao.save_mom_fib, {}),
            (self.calculate_obv, self.indicator_dao.save_obv, {}),
            (self.calculate_roc_fib, self.indicator_dao.save_roc_fib, {}),
            (self.calculate_amount_roc_fib, self.indicator_dao.save_amount_roc_fib, {}),
            (self.calculate_rsi_fib, self.indicator_dao.save_rsi_fib, {}),
            (self.calculate_sar, self.indicator_dao.save_sar, {}),
            (self.calculate_vroc_fib, self.indicator_dao.save_vroc_fib, {}),
            (self.calculate_vwap, self.indicator_dao.save_vwap, {}),
            (self.calculate_wr_fib, self.indicator_dao.save_wr_fib, {}),
            (self.calculate_sma_fib, self.indicator_dao.save_sma_fib, {}), # 假设 DAO 有 save_sma_fib
            (self.calculate_kc_fib, self.indicator_dao.save_kc_fib, {}),   # 假设 DAO 有 save_kc_fib
            (self.calculate_stoch_fib, self.indicator_dao.save_stoch_fib, {}), # 假设 DAO 有 save_stoch_fib
            (self.calculate_adl, self.indicator_dao.save_adl, {}),         # 假设 DAO 有 save_adl
            (self.calculate_pivot_points, self.indicator_dao.save_pivot_points, {}), # 假设 DAO 有 save_pivot_points
        ]
        for calc_func, save_func, params in indicator_tasks:
            indicator_name = calc_func.__name__.replace('calculate_', '').upper()
            # logger.debug(f"[{indicator_name}] 开始计算 for {stock_code} {time_level_str}")
            try:
                # 传递 ohlcv_df 的副本，确保每个计算函数拿到干净的数据
                # 因为 pandas-ta 可能会原地修改或添加列（虽然不常见）
                # 并且 VWAP 计算可能临时修改索引
                ohlcv_df_copy = ohlcv_df_raw.copy()
                indicator_result_df = calc_func(ohlcv_df_copy, **params)
                if indicator_result_df is not None and not indicator_result_df.empty:
                    # 确保结果的索引与原始数据对齐（或至少是其子集）
                    indicator_result_df = indicator_result_df.reindex(ohlcv_df_raw.index).dropna(how='all')
                    if not indicator_result_df.empty:
                        # logger.debug(f"[{indicator_name}] 计算完成 (结果行数: {len(indicator_result_df)}), 开始保存 for {stock_code} {time_level_str}")
                        # 保存所有计算出的非 NaN 结果
                        await save_func(stock_info, time_level_str, indicator_result_df) # 移除 dropna，让 DAO 处理
                    else:
                         logger.warning(f"[{indicator_name}] 计算结果在重索引后为空 for {stock_code} {time_level_str}")
                # else:
                #     logger.warning(f"[{indicator_name}] 计算结果为空或计算失败 for {stock_code} {time_level_str}")
            except Exception as e:
                logger.error(f"[{indicator_name}] 处理指标时发生严重错误 for {stock_code} {time_level_str}: {e}", exc_info=True)
        # logger.info(f"完成所有指标的计算和保存 for {stock_code} {time_level}")

    async def calculate_and_save_macd_indicators(self, stock_code: str, time_level: Union[TimeLevel, str]):
        """
        计算指定股票和时间级别的所有支持的指标，并保存到数据库。
        (已适配 pandas-ta 计算方法)
        """
        if ta is None:
            logger.error("pandas-ta 未加载，无法计算指标。请先安装 'pandas-ta'。")
            return
        # logger.info(f"开始计算和保存指标 for {stock_code} {time_level} using pandas-ta")
        stock_info = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock_info:
            logger.error(f"无法找到股票信息: {stock_code}，指标计算中止")
            return
        # 确定需要多少历史数据，取斐波那契最大值加上一些缓冲
        # 考虑 DMI/ADXR 等可能需要更多数据
        needed_bars = max(FIB_PERIODS) + 20 # 保持足够大的缓冲

        ohlcv_df_raw = await self._get_ohlcv_data(stock_code, time_level, needed_bars)
        if ohlcv_df_raw is None or ohlcv_df_raw.empty:
            logger.error(f"无法获取用于计算指标的历史数据 for {stock_code} {time_level}")
            return
        time_level_str = time_level.value if isinstance(time_level, TimeLevel) else str(time_level)
        # --- 逐个计算并保存 ---
        # 任务列表保持不变，因为函数签名和目的没变
        indicator_tasks = [
            (self.calculate_macd_fib, self.indicator_dao.save_macd_fib, {}), # 计算 MACD 和 EMA
            (self.calculate_rsi_fib, self.indicator_dao.save_rsi_fib, {}),
            (self.calculate_kdj_fib, self.indicator_dao.save_kdj_fib, {}), # 使用默认 M1=3, M2=3
            (self.calculate_boll, self.indicator_dao.save_boll, {}), # 使用默认参数 period=20
            (self.calculate_cci_fib, self.indicator_dao.save_cci_fib, {}),
            (self.calculate_mfi_fib, self.indicator_dao.save_mfi_fib, {}),
            (self.calculate_roc_fib, self.indicator_dao.save_roc_fib, {}),
            (self.calculate_dmi_fib, self.indicator_dao.save_dmi_fib, {}),
            (self.calculate_sar, self.indicator_dao.save_sar, {}),
            (self.calculate_amount_roc_fib, self.indicator_dao.save_amount_roc_fib, {}),
            (self.calculate_cmf_fib, self.indicator_dao.save_cmf_fib, {}),
            (self.calculate_obv, self.indicator_dao.save_obv, {}),
            (self.calculate_vwap, self.indicator_dao.save_vwap, {}),
        ]

        for calc_func, save_func, params in indicator_tasks:
            indicator_name = calc_func.__name__.replace('calculate_', '').upper()
            # logger.debug(f"[{indicator_name}] 开始计算 for {stock_code} {time_level_str}")
            try:
                # 传递 ohlcv_df 的副本，确保每个计算函数拿到干净的数据
                # 因为 pandas-ta 可能会原地修改或添加列（虽然不常见）
                # 并且 VWAP 计算可能临时修改索引
                ohlcv_df_copy = ohlcv_df_raw.copy()

                indicator_result_df = calc_func(ohlcv_df_copy, **params)

                if indicator_result_df is not None and not indicator_result_df.empty:
                    # 确保结果的索引与原始数据对齐（或至少是其子集）
                    indicator_result_df = indicator_result_df.reindex(ohlcv_df_raw.index).dropna(how='all')

                    if not indicator_result_df.empty:
                        # logger.debug(f"[{indicator_name}] 计算完成 (结果行数: {len(indicator_result_df)}), 开始保存 for {stock_code} {time_level_str}")
                        # 保存所有计算出的非 NaN 结果
                        await save_func(stock_info, time_level_str, indicator_result_df) # 移除 dropna，让 DAO 处理
                    else:
                         logger.warning(f"[{indicator_name}] 计算结果在重索引后为空 for {stock_code} {time_level_str}")

                # else:
                #     logger.warning(f"[{indicator_name}] 计算结果为空或计算失败 for {stock_code} {time_level_str}")
            except Exception as e:
                logger.error(f"[{indicator_name}] 处理指标时发生严重错误 for {stock_code} {time_level_str}: {e}", exc_info=True)

        # logger.info(f"完成所有指标的计算和保存 for {stock_code} {time_level}")

    async def prepare_strategy_dataframe(self, stock_code: str, timeframes: List[str],
        strategy_params: Dict[str, Any],
        limit_per_tf: int = 1200
    ) -> Optional[pd.DataFrame]:
        """
        准备增强版多时间周期策略所需的 DataFrame。
        此版本确保返回所有请求周期的 'close' 列，以及 volume_tf 周期的 'volume' 列。

        Args:
            stock_code (str): 股票代码.
            timeframes (List[str]): 策略需要的时间周期列表 (e.g., ['5', '15', '30', '60']).
                                     **调用者应确保此列表包含分析所需的周期 (如 T+0 分析的周期)**。
            strategy_params (Dict[str, Any]): 策略参数字典，需要包含相关指标周期和 'volume_tf'。
            limit_per_tf (int): 为每个时间周期和指标获取的最新记录数。

        Returns:
            Optional[pd.DataFrame]: 合并后的 DataFrame，索引为 trade_time (升序)，
                                     包含策略和分析所需的列。
                                     如果数据准备失败或不完整，则返回 None。
        """
        logger.info(f"[{stock_code}] 开始准备增强策略 DataFrame for timeframes: {timeframes}")

        # --- 1. 从策略参数中提取所需周期和配置 ---
        try:
            # 确保所有需要的参数都存在，否则返回 None 或抛出错误
            required_keys = [
                'rsi_period', 'kdj_period_k', 'volume_tf', 'cci_period',
                'mfi_period', 'roc_period', 'dmi_period', 'amount_ma_period',
                'cmf_period', 'obv_ma_period' #, 'ema_period' # ema_period 可能不需要，因为模型有多列
            ]
            missing_keys = [key for key in required_keys if key not in strategy_params]
            if missing_keys:
                logger.error(f"[{stock_code}] 策略参数 'strategy_params' 缺少必要的键: {missing_keys}")
                return None

            rsi_period = int(strategy_params['rsi_period'])
            kdj_period_k = int(strategy_params['kdj_period_k'])
            volume_tf = str(strategy_params['volume_tf']) # 量能确认和 VWAP 的周期
            cci_period = int(strategy_params['cci_period'])
            mfi_period = int(strategy_params['mfi_period'])
            roc_period = int(strategy_params['roc_period'])
            dmi_period = int(strategy_params['dmi_period'])
            amount_ma_period = int(strategy_params['amount_ma_period'])
            cmf_period = int(strategy_params['cmf_period'])
            obv_ma_period = int(strategy_params['obv_ma_period'])
            # ema_period = int(strategy_params.get('ema_period', 13)) # 假设默认用 13，或者 DAO 能处理

            # 确定所有需要获取数据的周期 (策略周期 + 量能周期)
            all_timeframes = set(timeframes)
            all_timeframes.add(volume_tf)
            all_timeframes_list = sorted(list(all_timeframes), key=int)

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"[{stock_code}] 处理策略参数时出错: {e}", exc_info=True)
            return None

        # --- 2. 创建并发任务列表 ---
        tasks = []
        task_descriptions = {} # 用于追踪任务

        # a) 获取所有需要周期的 OHLCV 数据 (用于 close, volume, high, low)
        for tf in all_timeframes_list:
            ohlcv_task = self.indicator_dao.get_history_ohlcv_df(stock_code, tf, limit=limit_per_tf)
            tasks.append(ohlcv_task)
            task_descriptions[len(tasks)-1] = {'type': 'ohlcv', 'tf': tf}

        # b) 获取所有策略时间周期的评分指标
        for tf in timeframes: # 注意：这里只循环策略需要的周期来获取指标
            # MACD
            macd_task = self.indicator_dao.get_macd_fib_df(stock_code, tf, limit_per_tf)
            tasks.append(macd_task)
            task_descriptions[len(tasks)-1] = {'type': 'macd', 'tf': tf}
            # RSI
            rsi_task = self.indicator_dao.get_rsi_fib_df(stock_code, tf, rsi_period, limit_per_tf)
            tasks.append(rsi_task)
            task_descriptions[len(tasks)-1] = {'type': 'rsi', 'tf': tf}
            # KDJ
            kdj_task = self.indicator_dao.get_kdj_fib_df(stock_code, tf, kdj_period_k, limit_per_tf)
            tasks.append(kdj_task)
            task_descriptions[len(tasks)-1] = {'type': 'kdj', 'tf': tf}
            # EMA (获取所有 FIB EMA 列) - 修改 DAO 调用方式
            # 假设 get_ema_fib_all_df 获取模型中所有 ema 列
            ema_task = self.indicator_dao.get_ema_fib_df(stock_code, tf, limit_per_tf)
            tasks.append(ema_task)
            task_descriptions[len(tasks)-1] = {'type': 'ema_all', 'tf': tf} # 标记为获取所有 EMA
            # BOLL
            boll_task = self.indicator_dao.get_boll_df(stock_code, tf, limit_per_tf)
            tasks.append(boll_task)
            task_descriptions[len(tasks)-1] = {'type': 'boll', 'tf': tf}
            # CCI
            cci_task = self.indicator_dao.get_cci_fib_df(stock_code, tf, cci_period, limit_per_tf)
            tasks.append(cci_task)
            task_descriptions[len(tasks)-1] = {'type': 'cci', 'tf': tf}
            # MFI
            mfi_task = self.indicator_dao.get_mfi_fib_df(stock_code, tf, mfi_period, limit_per_tf)
            tasks.append(mfi_task)
            task_descriptions[len(tasks)-1] = {'type': 'mfi', 'tf': tf}
            # ROC
            roc_task = self.indicator_dao.get_roc_fib_df(stock_code, tf, roc_period, limit_per_tf)
            tasks.append(roc_task)
            task_descriptions[len(tasks)-1] = {'type': 'roc', 'tf': tf}
            # DMI
            dmi_task = self.indicator_dao.get_dmi_fib_df(stock_code, tf, dmi_period, limit_per_tf)
            tasks.append(dmi_task)
            task_descriptions[len(tasks)-1] = {'type': 'dmi', 'tf': tf}
            # SAR
            sar_task = self.indicator_dao.get_sar_df(stock_code, tf, limit_per_tf)
            tasks.append(sar_task)
            task_descriptions[len(tasks)-1] = {'type': 'sar', 'tf': tf}

        # c) 获取 volume_tf 的量能确认指标
        # Amount MA
        amt_ma_task = self.indicator_dao.get_amount_ma_fib_df(stock_code, volume_tf, amount_ma_period, limit_per_tf)
        tasks.append(amt_ma_task)
        task_descriptions[len(tasks)-1] = {'type': 'amount_ma', 'tf': volume_tf}
        # CMF
        cmf_task = self.indicator_dao.get_cmf_fib_df(stock_code, volume_tf, cmf_period, limit_per_tf)
        tasks.append(cmf_task)
        task_descriptions[len(tasks)-1] = {'type': 'cmf', 'tf': volume_tf}
        # VWAP
        vwap_task = self.indicator_dao.get_vwap_df(stock_code, volume_tf, limit_per_tf)
        tasks.append(vwap_task)
        task_descriptions[len(tasks)-1] = {'type': 'vwap', 'tf': volume_tf}
        # OBV (注意：OBV MA 需要在后面计算)
        obv_task = self.indicator_dao.get_obv_df(stock_code, volume_tf, limit_per_tf)
        tasks.append(obv_task)
        task_descriptions[len(tasks)-1] = {'type': 'obv', 'tf': volume_tf}

        # --- 3. 并发执行所有 DAO 查询 ---
        logger.debug(f"[{stock_code}] 开始并发执行 {len(tasks)} 个数据获取任务...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        logger.debug(f"[{stock_code}] 数据获取任务执行完毕.")

        # --- 4. 处理结果并重命名列 ---
        dfs_to_merge: List[pd.DataFrame] = []
        fetched_data_summary = {}
        # 用于后续计算 OBV MA 的 Series
        obv_vol_tf_series: Optional[pd.Series] = None

        for i, result in enumerate(results):
            desc = task_descriptions[i]
            data_type = desc['type']
            tf = desc['tf']
            data_key = f"{data_type}_{tf}"
            required_cols_map = {} # 当前任务需要重命名的列

            if isinstance(result, Exception):
                logger.warning(f"[{stock_code}] 获取 {data_key} 数据时出错: {result}", exc_info=False)
                fetched_data_summary[data_key] = "Error"
                continue
            if result is None or result.empty:
                # logger.warning(f"[{stock_code}] 未获取到有效的 {data_key} 数据")
                fetched_data_summary[data_key] = "Empty/None"
                continue

            df = result.copy() # 操作副本

            # --- 定义重命名映射 ---
            if data_type == 'ohlcv':
                # 为所有周期的 OHLCV 数据提取 close 列
                required_cols_map = {'close': f'close_{tf}'}
                # 特别处理 volume_tf 周期，提取更多列
                if tf == volume_tf:
                    required_cols_map.update({
                        'high': f'high_{tf}',
                        'low': f'low_{tf}',
                        'volume': f'volume_{tf}', # 显式提取 volume
                        'turnover': f'amount_{tf}' # 保持 amount (来自 turnover)
                    })
            elif data_type == 'macd':
                required_cols_map = {'diff': f'diff_{tf}', 'dea': f'dea_{tf}', 'macd': f'macd_{tf}'}
            elif data_type == 'rsi':
                required_cols_map = {'rsi': f'rsi_{tf}'}
            elif data_type == 'kdj':
                required_cols_map = {'k': f'k_{tf}', 'd': f'd_{tf}', 'j': f'j_{tf}'}
            elif data_type == 'ema_all': # 处理获取所有 EMA 列的情况
                # 假设 DAO 返回的列名已经是 ema5, ema8, ... ema233
                # 我们需要将它们重命名为 ema5_tf, ema8_tf, ...
                required_cols_map = {f'ema{p}': f'ema{p}_{tf}' for p in FIB_PERIODS if f'ema{p}' in df.columns}
            elif data_type == 'boll':
                required_cols_map = {'upper': f'upper_{tf}', 'mid': f'mid_{tf}', 'lower': f'lower_{tf}'}
            elif data_type == 'cci':
                required_cols_map = {'cci': f'cci_{tf}'}
            elif data_type == 'mfi':
                required_cols_map = {'mfi': f'mfi_{tf}'}
            elif data_type == 'roc':
                required_cols_map = {'roc': f'roc_{tf}'}
            elif data_type == 'dmi':
                required_cols_map = {'pdi': f'pdi_{tf}', 'mdi': f'mdi_{tf}', 'adx': f'adx_{tf}'}
            elif data_type == 'sar':
                required_cols_map = {'sar': f'sar_{tf}'}
            elif data_type == 'amount_ma' and tf == volume_tf:
                required_cols_map = {'amount_ma': f'amount_ma_{tf}'}
            elif data_type == 'cmf' and tf == volume_tf:
                required_cols_map = {'cmf': f'cmf_{tf}'}
            elif data_type == 'vwap' and tf == volume_tf:
                required_cols_map = {'vwap': f'vwap_{tf}'}
            elif data_type == 'obv' and tf == volume_tf:
                required_cols_map = {'obv': f'obv_{tf}'}
                if 'obv' in df.columns:
                    # 提取 OBV Series 用于后续计算 MA
                    obv_vol_tf_series = df['obv'].copy()
                    # 确保索引是 DatetimeIndex
                    if not isinstance(obv_vol_tf_series.index, pd.DatetimeIndex):
                        try:
                            obv_vol_tf_series.index = pd.to_datetime(obv_vol_tf_series.index)
                        except Exception as e_idx:
                             logger.warning(f"[{stock_code}] 无法转换 OBV 索引为 DatetimeIndex: {e_idx}")
                             obv_vol_tf_series = None # 无法使用

            # --- 执行重命名并检查 ---
            try:
                # 检查原始列是否存在
                missing_original_cols = [orig_col for orig_col in required_cols_map.keys() if orig_col not in df.columns]
                if missing_original_cols:
                    logger.warning(f"[{stock_code}] 获取的 {data_key} 数据缺少原始列: {missing_original_cols}. 可用列: {df.columns.tolist()}")
                    # 移除缺失的映射，继续处理存在的列
                    required_cols_map = {k: v for k, v in required_cols_map.items() if k in df.columns}
                    if not required_cols_map: # 如果所有需要的列都缺失
                        fetched_data_summary[data_key] = "Missing All Original Cols"
                        continue # 跳过这个 DataFrame

                df.rename(columns=required_cols_map, inplace=True)
                renamed_cols = list(required_cols_map.values())

                # 再次检查重命名后的列是否存在
                actual_cols = df.columns.tolist()
                final_cols_to_keep = [col for col in renamed_cols if col in actual_cols]
                if len(final_cols_to_keep) != len(renamed_cols):
                     missing_renamed = set(renamed_cols) - set(final_cols_to_keep)
                     logger.warning(f"[{stock_code}] 重命名 {data_key} 列后，部分预期列丢失: {missing_renamed}. 保留列: {final_cols_to_keep}")
                     if not final_cols_to_keep: # 如果重命名后没有留下任何需要的列
                         fetched_data_summary[data_key] = "Rename Failed"
                         continue

                df_to_add = df[final_cols_to_keep]

                # 确保索引是 DatetimeIndex
                if not isinstance(df_to_add.index, pd.DatetimeIndex):
                    try:
                        df_to_add.index = pd.to_datetime(df_to_add.index)
                        # 确保时区（假设 DAO 返回的是 UTC 或无时区）
                        if df_to_add.index.tz is None:
                            df_to_add.index = df_to_add.index.tz_localize('UTC') # 或者 settings.TIME_ZONE
                        df_to_add.index = df_to_add.index.tz_convert(timezone.get_default_timezone()) # 转为 Django 默认时区
                    except Exception as e_idx:
                        logger.warning(f"[{stock_code}] 无法将 {data_key} 的索引转换为 DatetimeIndex 或处理时区: {e_idx}")
                        fetched_data_summary[data_key] = "Index Error"
                        continue # 跳过这个 DataFrame

                dfs_to_merge.append(df_to_add)
                fetched_data_summary[data_key] = "Success"

            except Exception as e_rename:
                logger.error(f"[{stock_code}] 处理或重命名 {data_key} 列时出错: {e_rename}", exc_info=True)
                fetched_data_summary[data_key] = "Processing Exception"

        # logger.info(f"[{stock_code}] 数据获取与初步处理概要: {fetched_data_summary}")

        # --- 5. 计算 OBV MA ---
        if ta is not None and obv_vol_tf_series is not None and not obv_vol_tf_series.empty:
            try:
                # 确保 OBV Series 索引是 DatetimeIndex 且有时区
                if not isinstance(obv_vol_tf_series.index, pd.DatetimeIndex):
                     raise ValueError("OBV Series index is not DatetimeIndex")
                if obv_vol_tf_series.index.tz is None:
                     obv_vol_tf_series.index = obv_vol_tf_series.index.tz_localize('UTC').tz_convert(timezone.get_default_timezone())

                obv_ma_series = ta.sma(obv_vol_tf_series, length=obv_ma_period)
                if obv_ma_series is not None and not obv_ma_series.empty:
                    obv_ma_df = obv_ma_series.to_frame(name=f'obv_ma_{volume_tf}')
                    # 索引应该已经处理好了
                    dfs_to_merge.append(obv_ma_df)
                    logger.debug(f"[{stock_code}] 成功计算并准备合并 obv_ma_{volume_tf}")
                else:
                    logger.warning(f"[{stock_code}] 计算 OBV MA (period={obv_ma_period}) 失败或返回空")
            except Exception as e_obv_ma:
                logger.error(f"[{stock_code}] 计算 OBV MA 时出错: {e_obv_ma}", exc_info=True)
        elif ta is None:
             logger.warning(f"[{stock_code}] pandas_ta 未安装，无法计算 OBV MA。")


        # --- 6. 合并 DataFrame ---
        if not dfs_to_merge:
            logger.error(f"[{stock_code}] 没有成功获取或处理任何有效的数据用于合并。")
            return None

        try:
            logger.debug(f"[{stock_code}] 开始合并 {len(dfs_to_merge)} 个 DataFrame...")
            # 确保所有 DataFrame 都有相同的时区或无时区以进行合并
            target_tz_final = timezone.get_default_timezone()
            processed_dfs = []
            for df_part in dfs_to_merge:
                if isinstance(df_part.index, pd.DatetimeIndex):
                    if df_part.index.tz is None:
                        df_part.index = df_part.index.tz_localize('UTC').tz_convert(target_tz_final)
                    elif df_part.index.tz != target_tz_final:
                        df_part.index = df_part.index.tz_convert(target_tz_final)
                    processed_dfs.append(df_part)
                else:
                    logger.warning(f"[{stock_code}] 合并前发现非 DatetimeIndex 的 DataFrame，已跳过。")

            if not processed_dfs:
                 logger.error(f"[{stock_code}] 没有有效的 DataFrame 可供合并。")
                 return None

            # 使用 concat 进行合并，它能更好地处理基于索引的对齐
            merged_df = pd.concat(processed_dfs, axis=1, join='outer') # outer join 保留所有时间点
            merged_df.sort_index(ascending=True, inplace=True)

            # 填充 NaN 值 (向前填充通常是合理的策略，表示沿用上一时间点的值)
            merged_df.ffill(inplace=True)

            # 清理：删除那些关键列仍然是 NaN 的行
            # 选择一个必须存在的列作为基准，例如评分策略主要依赖的周期的 close 列
            # 或者，如果 volume_tf 的 close 列更可靠，用它
            key_col = f'close_{volume_tf}' # 或者选择一个策略中最常用的 tf
            if key_col not in merged_df.columns:
                 # 如果 volume_tf 的 close 列不存在，尝试用第一个 timeframes 的 close
                 if timeframes and f'close_{timeframes[0]}' in merged_df.columns:
                      key_col = f'close_{timeframes[0]}'
                 else:
                      # 如果连第一个周期的 close 都没有，可能数据有问题，但尝试继续
                      key_col = None
                      logger.warning(f"[{stock_code}] 无法找到合适的关键列用于 dropna，可能保留过多无效行。")

            if key_col:
                initial_rows = len(merged_df)
                # 删除关键列为 NaN 的行，这些行通常是数据获取不完整的开始部分
                merged_df.dropna(subset=[key_col], inplace=True)
                logger.debug(f"[{stock_code}] 基于关键列 '{key_col}' 清理后，行数从 {initial_rows} 变为 {len(merged_df)}")

            # 最后再检查一次是否为空
            if merged_df.empty:
                logger.error(f"[{stock_code}] 合并并清理后 DataFrame 为空。")
                return None

            logger.info(f"[{stock_code}] 增强策略 DataFrame 准备完成，最终形状: {merged_df.shape}")
            # logger.debug(f"[{stock_code}] 最终列名: {merged_df.columns.tolist()}")
            return merged_df

        except Exception as e_merge:
            logger.error(f"[{stock_code}] 合并或后处理 DataFrame 时出错: {e_merge}", exc_info=True)
            return None














