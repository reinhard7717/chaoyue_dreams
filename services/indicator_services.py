import asyncio
from collections import defaultdict
from functools import reduce
import json
import os
import warnings
import logging
import pandas as pd
import numpy as np
from django.utils import timezone
from typing import Any, List, Optional, Type, Union, Dict, Tuple
from django.db import models # 确保导入 models
import pandas_ta as ta

from stock_models.indicator.adl import StockAdl
from stock_models.indicator.atr import StockAtrFIB
from stock_models.indicator.boll import StockBOLLIndicator
from stock_models.indicator.cci import StockCciFIB
from stock_models.indicator.cmf import StockCmfFIB
from stock_models.indicator.dmi import StockDmiFIB
from stock_models.indicator.ichimoku import StockIchimoku
from stock_models.indicator.kc import StockKcFIB
from stock_models.indicator.kdj import StockKDJFIB
from stock_models.indicator.ma import StockAmountMaFIB, StockEmaFIB
from stock_models.indicator.macd import StockMACDFIB
from stock_models.indicator.mfi import StockMfiFIB
from stock_models.indicator.mom import StockMomFIB
from stock_models.indicator.obv import StockObvFIB
from stock_models.indicator.pivot_points import StockPivotPoints
from stock_models.indicator.roc import StockAmountRocFIB, StockRocFIB
from stock_models.indicator.rsi import StockRsiFIB
from stock_models.indicator.sar import StockSar
from stock_models.indicator.sma import StockSmaFIB
from stock_models.indicator.stochastic_oscillator import StockStochFIB
from stock_models.indicator.vroc import StockVrocFIB
from stock_models.indicator.vwap import StockVwap
from stock_models.indicator.wr import StockWrFIB

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
            if ta is None: # 如果之前导入失败过
                 logger.warning("pandas_ta 之前导入失败，尝试重新导入。")
                 import pandas_ta as ta
            # 添加 ta.CommonStrategy 以便后面使用
            # if not hasattr(ta, 'CommonStrategy'):
            #     # 根据 pandas_ta 版本，可能需要导入或它已在 ta 命名空间下
            #     pass # 通常 ta.Strategy 即可
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
        # --- 添加：返回前，确保必要的列是 float 类型 ---
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        if 'turnover' in df.columns: ohlcv_cols.append('turnover')
        for col in ohlcv_cols:
             if col in df.columns:
                 try:
                      # 使用 errors='coerce' 将无效解析转为 NaN
                      df[col] = pd.to_numeric(df[col], errors='coerce')
                 except Exception as e:
                      logger.warning(f"无法将列 {col} 转换为数值类型: {e}")
        # --- 确保索引是 DatetimeIndex 且时区感知 ---
        if not isinstance(df.index, pd.DatetimeIndex):
             logger.warning(f"OHLCV 数据索引不是 DatetimeIndex，尝试转换...")
             try:
                 df.index = pd.to_datetime(df.index)
             except Exception as e_idx:
                 logger.error(f"无法将 OHLCV 索引转换为 DatetimeIndex: {e_idx}")
                 return None # 无法继续

        default_tz = timezone.get_default_timezone()
        if df.index.tz is None:
             logger.warning(f"OHLCV 数据索引是 naive 的，将本地化为默认时区...")
             try:
                 df.index = df.index.tz_localize(default_tz)
             except Exception as e_tz:
                 logger.error(f"无法本地化 OHLCV 索引时区: {e_tz}")
                 return None # 无法继续
        elif df.index.tz != default_tz:
             # 如果已经是 aware 但时区不同，转换为默认时区
             df.index = df.index.tz_convert(default_tz)
        # --- 时区处理结束 ---
        
        return df

    # --- 单个指标计算方法 (使用 pandas-ta) ---

    def calculate_atr(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 ATR。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period (int): ATR 计算周期。
        Returns:
            Optional[pd.DataFrame]: 包含 'ATR_period' 列的 DataFrame，如果计算失败则返回 None。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < period: return None
        try:
            atr_series = ohlc.ta.atr(length=period)
            if atr_series is None: return None
            # 返回 DataFrame，列名为 ATR_period
            return atr_series.to_frame(name=f'ATR_{period}')
        except Exception as e:
            logger.error(f"计算 ATR(period={period}) 失败: {e}", exc_info=True)
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

    def calculate_cci(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 CCI。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period (int): CCI 计算周期。
        Returns:
            Optional[pd.DataFrame]: 包含 'CCI_period' 列的 DataFrame，如果计算失败则返回 None。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < period: return None
        try:
            cci_series = ohlc.ta.cci(length=period)
            if cci_series is None: return None
            # 返回 DataFrame，列名为 CCI_period
            return cci_series.to_frame(name=f'CCI_{period}')
        except Exception as e:
            logger.error(f"计算 CCI(period={period}) 失败: {e}", exc_info=True)
            return None
        
    def calculate_cmf(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 CMF。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据 (必须包含 'volume' 列)。
            period (int): CMF 计算周期。
        Returns:
            Optional[pd.DataFrame]: 包含 'CMF_period' 列的 DataFrame，如果计算失败则返回 None。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or 'volume' not in ohlc.columns:
            logger.error("计算 CMF 需要 'volume' 列")
            return None
        if len(ohlc) < period: return None
        try:
            cmf_series = ohlc.ta.cmf(length=period)
            if cmf_series is None: return None
            # 返回 DataFrame，列名为 CMF_period
            return cmf_series.to_frame(name=f'CMF_{period}')
        except Exception as e:
            logger.error(f"计算 CMF(period={period}) 失败: {e}", exc_info=True)
            return None

    def calculate_dmi(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 DMI (+DI, -DI, ADX)。
        注意：不直接计算 ADXR。

        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period (int): DMI 计算周期。

        Returns:
            Optional[pd.DataFrame]: 包含 '+DI_period', '-DI_period', 'ADX_period' 列的 DataFrame。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        # ADX 通常需要至少 2*period 的数据
        if ohlc is None or ohlc.empty or len(ohlc) < period * 2: return None
        try:
            adx_df = ohlc.ta.adx(length=period)
            if adx_df is None or adx_df.empty: return None

            # 提取所需的列并重命名
            rename_map = {}
            pdi_col_ta = f'DMP_{period}'
            mdi_col_ta = f'DMN_{period}'
            adx_col_ta = f'ADX_{period}'
            pdi_col_strat = f'+DI_{period}'
            mdi_col_strat = f'-DI_{period}'
            adx_col_strat = f'ADX_{period}' # 保持 ADX 名称

            required_cols = [pdi_col_ta, mdi_col_ta, adx_col_ta]
            found_cols = {}
            if pdi_col_ta in adx_df.columns: found_cols[pdi_col_ta] = pdi_col_strat
            if mdi_col_ta in adx_df.columns: found_cols[mdi_col_ta] = mdi_col_strat
            if adx_col_ta in adx_df.columns: found_cols[adx_col_ta] = adx_col_strat

            if len(found_cols) < 3:
                 logger.warning(f"无法从 pandas-ta adx(length={period}) 结果中找到所有 PDI/MDI/ADX 列。返回列: {adx_df.columns.tolist()}")
                 # 尝试返回找到的列
                 if not found_cols: return None
                 return adx_df[list(found_cols.keys())].rename(columns=found_cols)

            return adx_df[list(found_cols.keys())].rename(columns=found_cols)

        except Exception as e:
            logger.error(f"计算 DMI(period={period}) 失败: {e}", exc_info=True)
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

    def calculate_kdj(self, ohlc: pd.DataFrame, period: int, signal_period: int, smooth_k_period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 KDJ (K, D, J)。
        【修改】: 返回列名为 K_period_signal, D_period_signal, J_period_signal

        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period (int): K 计算周期 (N)。
            signal_period (int): D 计算周期 (M1)。
            smooth_k_period (int): K 平滑周期 (M2)。

        Returns:
            Optional[pd.DataFrame]: 包含 'K_period_signal', 'D_period_signal', 'J_period_signal' 列。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < period + signal_period + smooth_k_period: return None
        try:
            kdj_df = ohlc.ta.kdj(length=period, signal=signal_period, smooth_k=smooth_k_period)
            if kdj_df is None or kdj_df.empty: return None
            # 构建 pandas-ta 可能的列名
            k_col_ta = f'K_{period}_{signal_period}'
            d_col_ta = f'D_{period}_{signal_period}'
            j_col_ta = f'J_{period}_{signal_period}'
            # 策略期望的基础列名 (不带时间后缀)
            k_col_base = f'K_{period}_{signal_period}'
            d_col_base = f'D_{period}_{signal_period}'
            j_col_base = f'J_{period}_{signal_period}'
            rename_map = {}
            if k_col_ta in kdj_df.columns: rename_map[k_col_ta] = k_col_base
            if d_col_ta in kdj_df.columns: rename_map[d_col_ta] = d_col_base
            if j_col_ta in kdj_df.columns: rename_map[j_col_ta] = j_col_base
            # 回退到位置索引
            if len(rename_map) < 3:
                logger.warning(f"KDJ 列名不匹配预期。尝试位置索引。返回列: {kdj_df.columns.tolist()}")
                if k_col_base not in rename_map.values() and len(kdj_df.columns) >= 1:
                     # 检查原始列名是否已经是期望的基础名
                     if kdj_df.columns[0] == k_col_base:
                         rename_map[kdj_df.columns[0]] = k_col_base
                     else: # 否则用位置索引重命名
                         rename_map[kdj_df.columns[0]] = k_col_base
                if d_col_base not in rename_map.values() and len(kdj_df.columns) >= 2:
                     if kdj_df.columns[1] == d_col_base:
                          rename_map[kdj_df.columns[1]] = d_col_base
                     else:
                          rename_map[kdj_df.columns[1]] = d_col_base
                if j_col_base not in rename_map.values() and len(kdj_df.columns) >= 3:
                     if kdj_df.columns[2] == j_col_base:
                          rename_map[kdj_df.columns[2]] = j_col_base
                     else:
                          rename_map[kdj_df.columns[2]] = j_col_base
            if len(rename_map) < 3:
                 logger.error(f"无法从 pandas-ta kdj 结果中提取 K, D, J 列。找到的映射: {rename_map}")
                 # 只返回能找到的列，并使用期望的基础名称
                 if not rename_map: return None
                 return kdj_df[[col for col in rename_map.keys()]].rename(columns=rename_map)
            # 返回包含 K_period_signal, D_period_signal, J_period_signal 的 DataFrame
            return kdj_df[[col for col in rename_map.keys()]].rename(columns=rename_map)
        except Exception as e:
            logger.error(f"计算 KDJ(k={period}, s={signal_period}, sm={smooth_k_period}) 失败: {e}", exc_info=True)
            return None
        
    def calculate_ema(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 EMA。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period (int): EMA 计算周期。
        Returns:
            Optional[pd.DataFrame]: 包含 'EMA_period' 列的 DataFrame，如果计算失败则返回 None。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < period: return None
        try:
            # 假设计算收盘价的 EMA
            ema_series = ohlc.ta.ema(close='close', length=period) # 明确指定 close='close'
            if ema_series is None: return None
            # 返回 DataFrame，列名为 EMA_period
            return ema_series.to_frame(name=f'EMA_{period}')
        except Exception as e:
            logger.error(f"计算 EMA(period={period}) 失败: {e}", exc_info=True)
            return None
        
    def calculate_amount_ma(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的成交额 SMA。

        Args:
            ohlc (pd.DataFrame): OHLCV 数据 (必须包含 'turnover' 列)。
            period (int): SMA 计算周期。

        Returns:
            Optional[pd.DataFrame]: 包含 'AMT_MA_period' 列的 DataFrame，如果计算失败则返回 None。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or 'turnover' not in ohlc.columns:
            logger.error("计算 Amount MA 需要 'turnover' 列")
            return None
        if len(ohlc) < period: return None
        try:
            amt_ma_series = ohlc.ta.sma(close='turnover', length=period)
            if amt_ma_series is None: return None
            # 返回 DataFrame，列名为 AMT_MA_period
            return amt_ma_series.to_frame(name=f'AMT_MA_{period}')
        except Exception as e:
            logger.error(f"计算 Amount MA(period={period}) 失败: {e}", exc_info=True)
            return None

    def calculate_macd(self, ohlc: pd.DataFrame, period_fast: int = 12, period_slow: int = 26, signal_period: int = 9) -> Optional[pd.DataFrame]:
        """
        计算标准 MACD(fast, slow, signal)。
        返回包含 'MACD_f_s_g', 'MACDh_f_s_g', 'MACDs_f_s_g' 列的 DataFrame。

        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period_fast (int): 快线 EMA 周期。
            period_slow (int): 慢线 EMA 周期。
            signal_period (int): 信号线 EMA 周期。

        Returns:
            Optional[pd.DataFrame]: 包含 MACD, MACD Histogram, MACD Signal 列。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < period_slow + signal_period: return None
        try:
            # 直接调用 pandas-ta macd
            macd_df = ohlc.ta.macd(fast=period_fast, slow=period_slow, signal=signal_period)

            if macd_df is None or macd_df.empty:
                logger.warning(f"pandas-ta macd({period_fast}, {period_slow}, {signal_period}) 计算失败或返回空")
                return None
            # 不再进行重命名，直接返回 pandas-ta 的结果
            # 列名将是 'MACD_f_s_g', 'MACDh_f_s_g', 'MACDs_f_s_g'
            return macd_df

        except Exception as e:
            logger.error(f"计算 MACD({period_fast}, {period_slow}, {signal_period}) 失败: {e}", exc_info=True)
            return None

    def calculate_mfi(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 MFI。

        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period (int): MFI 计算周期。

        Returns:
            Optional[pd.DataFrame]: 包含 'MFI_period' 列的 DataFrame，如果计算失败则返回 None。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < period: return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                mfi_series = ohlc.ta.mfi(length=period)
            if mfi_series is None: return None
            # 返回 DataFrame，列名为 MFI_period
            return mfi_series.to_frame(name=f'MFI_{period}')
        except Exception as e:
            logger.error(f"计算 MFI(period={period}) 失败: {e}", exc_info=True)
            return None

    def calculate_mom(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 MOM。

        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period (int): MOM 计算周期。

        Returns:
            Optional[pd.DataFrame]: 包含 'MOM_period' 列的 DataFrame，如果计算失败则返回 None。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < period: return None
        try:
            mom_series = ohlc.ta.mom(length=period)
            if mom_series is None: return None
            # 返回 DataFrame，列名为 MOM_period
            return mom_series.to_frame(name=f'MOM_{period}')
        except Exception as e:
            logger.error(f"计算 MOM(period={period}) 失败: {e}", exc_info=True)
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

    def calculate_roc(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 ROC。

        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period (int): ROC 计算周期。

        Returns:
            Optional[pd.DataFrame]: 包含 'ROC_period' 列的 DataFrame，如果计算失败则返回 None。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < period + 1: return None
        try:
            roc_series = ohlc.ta.roc(length=period)
            if roc_series is None: return None
            # 返回 DataFrame，列名为 ROC_period
            df_result = roc_series.to_frame(name=f'ROC_{period}')
            df_result.replace([np.inf, -np.inf], np.nan, inplace=True) # 处理 inf
            return df_result
        except Exception as e:
            logger.error(f"计算 ROC(period={period}) 失败: {e}", exc_info=True)
            return None

    def calculate_amount_roc(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的成交额 ROC。

        Args:
            ohlc (pd.DataFrame): OHLCV 数据 (必须包含 'turnover' 列)。
            period (int): ROC 计算周期。

        Returns:
            Optional[pd.DataFrame]: 包含 'AROC_period' 列的 DataFrame，如果计算失败则返回 None。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or 'turnover' not in ohlc.columns:
             logger.error("计算 Amount ROC 需要 'turnover' 列")
             return None
        if len(ohlc) < period + 1: return None
        try:
            aroc_series = ohlc.ta.roc(close='turnover', length=period)
            if aroc_series is None: return None
            # 返回 DataFrame，列名为 AROC_period
            df_results = aroc_series.to_frame(name=f'AROC_{period}')
            df_results.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_results
        except Exception as e:
            logger.error(f"计算 Amount ROC(period={period}) 失败: {e}", exc_info=True)
            return None

    def calculate_rsi(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 RSI。

        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period (int): RSI 计算周期。

        Returns:
            Optional[pd.DataFrame]: 包含 'RSI_period' 列的 DataFrame，如果计算失败则返回 None。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < period + 1: return None
        try:
            rsi_series = ohlc.ta.rsi(length=period)
            if rsi_series is None: return None
            # 返回 DataFrame，列名为 RSI_period
            return rsi_series.to_frame(name=f'RSI_{period}')
        except Exception as e:
            logger.error(f"计算 RSI(period={period}) 失败: {e}", exc_info=True)
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

    def calculate_vroc(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的成交量 ROC (VROC)。

        Args:
            ohlc (pd.DataFrame): OHLCV 数据 (必须包含 'volume' 列)。
            period (int): VROC 计算周期。

        Returns:
            Optional[pd.DataFrame]: 包含 'VROC_period' 列的 DataFrame，如果计算失败则返回 None。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or 'volume' not in ohlc.columns:
             logger.error("计算 VROC 需要 'volume' 列")
             return None
        if len(ohlc) < period + 1: return None
        try:
            vroc_series = ohlc.ta.roc(close='volume', length=period)
            if vroc_series is None: return None
            # 返回 DataFrame，列名为 VROC_period
            df_results = vroc_series.to_frame(name=f'VROC_{period}')
            df_results.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_results
        except Exception as e:
            logger.error(f"计算 VROC(period={period}) 失败: {e}", exc_info=True)
            return None

    def calculate_vwap(self, ohlc: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        计算 VWAP (日内 VWAP 需要 timestamp index)。

        Args:
            ohlc (pd.DataFrame): OHLCV 数据。

        Returns:
            Optional[pd.DataFrame]: 包含 'VWAP' 列的 DataFrame，如果计算失败则返回 None。
                                     列名通常由 pandas_ta 决定 (e.g., VWAP_D)，会尝试重命名为 'VWAP'。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty: return None
        # pandas-ta vwap 依赖于 DatetimeIndex 来确定重置周期 (通常是每日)
        # 假定输入 ohlc 的索引已经是 DatetimeIndex (在 _get_ohlcv_data 中处理)
        if not isinstance(ohlc.index, pd.DatetimeIndex):
            logger.warning("计算 VWAP 需要 DataFrame 的 index 是 DatetimeIndex。尝试转换...")
            try:
                original_index = ohlc.index # 保存原始索引以备恢复
                ohlc.index = pd.to_datetime(ohlc.index)
            except Exception as e:
                logger.error(f"无法将 index 转换为 DatetimeIndex，VWAP 计算可能不准确或失败: {e}")
                # 可以在这里决定是否返回 None 或继续尝试
                # return None
        try:
            # pandas-ta vwap 返回 Series，名称通常是 VWAP_D (日级别) 或类似
            vwap_series = ohlc.ta.vwap() # 默认计算日 VWAP
            if vwap_series is None:
                 logger.warning("pandas-ta vwap() 返回 None")
                 return None
            # 尝试将返回的 Series 重命名为 'VWAP'
            # 获取 pandas_ta 返回的实际列名 (通常只有一个)
            vwap_col_name = vwap_series.name if hasattr(vwap_series, 'name') else None
            if vwap_col_name:
                 return vwap_series.to_frame(name='VWAP')
            else:
                 # 如果没有名称，直接创建一个名为 VWAP 的列
                 logger.warning("pandas-ta vwap() 返回的 Series 没有名称，将尝试直接创建 'VWAP' 列")
                 return pd.DataFrame({'VWAP': vwap_series}, index=ohlc.index)
        except Exception as e:
            logger.error(f"计算 VWAP 失败: {e}", exc_info=True)
            return None
        finally:
            # 恢复索引 (如果之前转换过)
            if 'original_index' in locals():
                ohlc.index = original_index
                pass

    def calculate_wr(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 Williams %R (WR)。

        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period (int): WR 计算周期。

        Returns:
            Optional[pd.DataFrame]: 包含 'WR_period' 列的 DataFrame，如果计算失败则返回 None。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < period: return None
        try:
            wr_series = ohlc.ta.willr(length=period)
            if wr_series is None: return None
            # 返回 DataFrame，列名为 WR_period
            return wr_series.to_frame(name=f'WR_{period}')
        except Exception as e:
            logger.error(f"计算 WR(period={period}) 失败: {e}", exc_info=True)
            return None

    def calculate_sma(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 SMA。

        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period (int): SMA 计算周期。

        Returns:
            Optional[pd.DataFrame]: 包含 'SMA_period' 列的 DataFrame，如果计算失败则返回 None。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < period: return None
        try:
            sma_series = ohlc.ta.sma(length=period)
            if sma_series is None: return None
            # 返回 DataFrame，列名为 SMA_period
            return sma_series.to_frame(name=f'SMA_{period}')
        except Exception as e:
            logger.error(f"计算 SMA(period={period}) 失败: {e}", exc_info=True)
            return None

    def calculate_kc(self, ohlc: pd.DataFrame, period: int, atr_length: int = 10, scalar: float = 2.0) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 Keltner Channels (KC)。
        使用 period 作为 EMA 的长度 (length)。

        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period (int): EMA 计算周期。
            atr_length (int): ATR 计算周期。
            scalar (float): ATR 乘数。

        Returns:
            Optional[pd.DataFrame]: 包含 'KC_LOWER_period', 'KC_BASIS_period', 'KC_UPPER_period' 列。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < max(period, atr_length): return None
        try:
            kc_df = ohlc.ta.kc(length=period, atr_length=atr_length, scalar=scalar)
            if kc_df is None or kc_df.empty: return None

            # 提取并重命名
            lower_col_ta = f'KCLe_{period}_{scalar}'
            basis_col_ta = f'KCBe_{period}_{scalar}'
            upper_col_ta = f'KCUe_{period}_{scalar}'
            lower_col_strat = f'KC_LOWER_{period}'
            basis_col_strat = f'KC_BASIS_{period}'
            upper_col_strat = f'KC_UPPER_{period}'

            found_cols = {}
            if lower_col_ta in kc_df.columns: found_cols[lower_col_ta] = lower_col_strat
            if basis_col_ta in kc_df.columns: found_cols[basis_col_ta] = basis_col_strat
            if upper_col_ta in kc_df.columns: found_cols[upper_col_ta] = upper_col_strat

             # 回退到位置索引
            if len(found_cols) < 3:
                 logger.warning(f"pandas-ta kc(p={period}, atr={atr_length}, s={scalar}) 列名不完全匹配预期。尝试位置索引。返回列: {kc_df.columns.tolist()}")
                 if lower_col_strat not in found_cols and len(kc_df.columns) >= 1:
                      found_cols[kc_df.columns[0]] = lower_col_strat
                 if basis_col_strat not in found_cols and len(kc_df.columns) >= 2:
                      found_cols[kc_df.columns[1]] = basis_col_strat
                 if upper_col_strat not in found_cols and len(kc_df.columns) >= 3:
                      found_cols[kc_df.columns[2]] = upper_col_strat

            if len(found_cols) < 3:
                 logger.error(f"无法从 pandas-ta kc 结果中提取 Lower, Basis, Upper 列。")
                 if not found_cols: return None
                 return kc_df[[col for col in found_cols.keys()]].rename(columns=found_cols)

            return kc_df[list(found_cols.keys())].rename(columns=found_cols)

        except Exception as e:
            logger.error(f"计算 KC(p={period}, atr={atr_length}, s={scalar}) 失败: {e}", exc_info=True)
            return None

    def calculate_stoch(self, ohlc: pd.DataFrame, k_period: int, d_period: int = 3, smooth_k_period: int = 3) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 Stochastic Oscillator (%K, %D)。
        使用 k_period 作为 %K 的计算周期 (k)。

        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            k_period (int): %K 计算周期。
            d_period (int): %D 计算周期。
            smooth_k_period (int): Slow %K 平滑周期。

        Returns:
            Optional[pd.DataFrame]: 包含 'STOCH_K_k_period', 'STOCH_D_k_period' 列。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < k_period + d_period + smooth_k_period: return None
        try:
            stoch_df = ohlc.ta.stoch(k=k_period, d=d_period, smooth_k=smooth_k_period)
            if stoch_df is None or stoch_df.empty: return None

            # 提取并重命名
            k_col_ta = f'STOCHk_{k_period}_{d_period}_{smooth_k_period}'
            d_col_ta = f'STOCHd_{k_period}_{d_period}_{smooth_k_period}'
            k_col_strat = f'STOCH_K_{k_period}' # 策略期望的列名
            d_col_strat = f'STOCH_D_{k_period}' # 策略期望的列名 (基于 K 周期命名)

            found_cols = {}
            if k_col_ta in stoch_df.columns: found_cols[k_col_ta] = k_col_strat
            if d_col_ta in stoch_df.columns: found_cols[d_col_ta] = d_col_strat

            if len(found_cols) < 2:
                logger.warning(f"pandas-ta stoch(k={k_period}, d={d_period}, sm={smooth_k_period}) 列名不完全匹配预期。返回列: {stoch_df.columns.tolist()}")
                if not found_cols: return None
                return stoch_df[[col for col in found_cols.keys()]].rename(columns=found_cols)

            return stoch_df[list(found_cols.keys())].rename(columns=found_cols)

        except Exception as e:
             logger.error(f"计算 Stochastic(k={k_period}, d={d_period}, sm={smooth_k_period}) 失败: {e}", exc_info=True)
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

    # --- 统一计算和保存入口  ---
    async def calculate_and_save_all_indicators(self, stock_code: str, time_level: Union[TimeLevel, str]):
        """
        计算指定股票和时间级别的所有支持的指标，并保存到数据库。
        优化：如果最新 K 线数据对应的指标已存在，则跳过该指标的计算和保存。
              否则，计算指标并保存结果中数据库尚不存在的数据。
        """
        if ta is None:
            logger.error("pandas-ta 未加载，无法计算指标。")
            return

        stock_info = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock_info:
            logger.error(f"无法找到股票信息: {stock_code}，指标计算中止")
            return

        time_level_str = time_level.value if isinstance(time_level, TimeLevel) else str(time_level)
        # logger.info(f"开始检查并计算/保存指标 for {stock_code} {time_level_str}")

        # 1. 获取基础 OHLCV 数据 (确保索引是时区感知的)
        needed_bars = max(FIB_PERIODS) + 50
        ohlcv_df_raw = await self._get_ohlcv_data(stock_code, time_level, needed_bars)
        if ohlcv_df_raw is None or ohlcv_df_raw.empty:
            logger.error(f"无法获取用于计算指标的历史数据 for {stock_code} {time_level_str}")
            return

        # 2. 获取最新 K 线的时间戳 (必须是时区感知的)
        latest_ohlcv_timestamp = ohlcv_df_raw.index[-1]
        # logger.debug(f"最新的 OHLCV 时间戳: {latest_ohlcv_timestamp} for {stock_code} {time_level_str}")

        # 获取所有需要检查的时间戳列表 (用于后续过滤保存)
        timestamps_in_ohlcv = list(ohlcv_df_raw.index)

        # --- indicator_tasks 列表保持不变 (包含模型类) ---
        indicator_tasks: List[Tuple[callable, callable, Type[models.Model], Dict]] = [
            (self.calculate_atr_fib, self.indicator_dao.save_atr_fib, StockAtrFIB, {}),
            (self.calculate_boll, self.indicator_dao.save_boll, StockBOLLIndicator, {}),
            (self.calculate_cci_fib, self.indicator_dao.save_cci_fib, StockCciFIB, {}),
            (self.calculate_cmf_fib, self.indicator_dao.save_cmf_fib, StockCmfFIB, {}),
            (self.calculate_dmi_fib, self.indicator_dao.save_dmi_fib, StockDmiFIB, {}),
            (self.calculate_ichimoku, self.indicator_dao.save_ichimoku, StockIchimoku, {}),
            (self.calculate_kdj_fib, self.indicator_dao.save_kdj_fib, StockKDJFIB, {}),
            (self.calculate_ema_fib, self.indicator_dao.save_ema_fib, StockEmaFIB, {}),
            (self.calculate_amount_ma_fib, self.indicator_dao.save_amount_ma_fib, StockAmountMaFIB, {}),
            (self.calculate_macd_fib, self.indicator_dao.save_macd_fib, StockMACDFIB, {}),
            (self.calculate_mfi_fib, self.indicator_dao.save_mfi_fib, StockMfiFIB, {}),
            (self.calculate_mom_fib, self.indicator_dao.save_mom_fib, StockMomFIB, {}),
            (self.calculate_obv, self.indicator_dao.save_obv, StockObvFIB, {}),
            (self.calculate_roc_fib, self.indicator_dao.save_roc_fib, StockRocFIB, {}),
            (self.calculate_amount_roc_fib, self.indicator_dao.save_amount_roc_fib, StockAmountRocFIB, {}),
            (self.calculate_rsi_fib, self.indicator_dao.save_rsi_fib, StockRsiFIB, {}),
            (self.calculate_sar, self.indicator_dao.save_sar, StockSar, {}),
            (self.calculate_vroc_fib, self.indicator_dao.save_vroc_fib, StockVrocFIB, {}),
            (self.calculate_vwap, self.indicator_dao.save_vwap, StockVwap, {}),
            (self.calculate_wr_fib, self.indicator_dao.save_wr_fib, StockWrFIB, {}),
            (self.calculate_sma_fib, self.indicator_dao.save_sma_fib, StockSmaFIB, {}),
            (self.calculate_kc_fib, self.indicator_dao.save_kc_fib, StockKcFIB, {}),
            (self.calculate_stoch_fib, self.indicator_dao.save_stoch_fib, StockStochFIB, {}),
            (self.calculate_adl, self.indicator_dao.save_adl, StockAdl, {}),
            (self.calculate_pivot_points, self.indicator_dao.save_pivot_points, StockPivotPoints, {}),
        ]

        # --- 修改循环逻辑 ---
        for calc_func, save_func, model_class, params in indicator_tasks:
            indicator_name = calc_func.__name__.replace('calculate_', '').upper()

            try:
                # 3. 检查最新时间戳的指标数据是否已存在
                latest_data_exists = await self.indicator_dao.check_indicator_exists_at_timestamp(
                    stock_info, time_level_str, model_class, latest_ohlcv_timestamp
                )

                if latest_data_exists:
                    # logger.debug(f"[{indicator_name}] 最新时间戳 {latest_ohlcv_timestamp} 的数据已存在，跳过计算和保存 for {stock_code} {time_level_str}")
                    continue # 跳到下一个指标

                # logger.debug(f"[{indicator_name}] 最新时间戳数据不存在，开始计算 for {stock_code} {time_level_str}")

                # 4. 计算指标 (如果最新数据不存在)
                ohlcv_df_copy = ohlcv_df_raw.copy()
                indicator_result_df = calc_func(ohlcv_df_copy, **params)

                if indicator_result_df is not None and not indicator_result_df.empty:
                    # 确保结果索引是时区感知的
                    if not isinstance(indicator_result_df.index, pd.DatetimeIndex):
                         indicator_result_df.index = pd.to_datetime(indicator_result_df.index)
                    default_tz = timezone.get_default_timezone()
                    if indicator_result_df.index.tz is None:
                         indicator_result_df.index = indicator_result_df.index.tz_localize(default_tz)
                    elif indicator_result_df.index.tz != default_tz:
                         indicator_result_df.index = indicator_result_df.index.tz_convert(default_tz)

                    # 5. 查询数据库中已存在的时间戳 (用于过滤保存)
                    existing_timestamps_set = await self.indicator_dao.get_existing_timestamps_for_range(
                        stock_info, time_level_str, model_class, timestamps_in_ohlcv
                    )

                    # 6. 过滤掉计算结果中时间戳已存在于数据库的行
                    indicator_result_df_filtered = indicator_result_df[~indicator_result_df.index.isin(existing_timestamps_set)]

                    # 7. 对齐索引并去除全 NaN 行 (可选)
                    indicator_result_df_filtered = indicator_result_df_filtered.reindex(ohlcv_df_raw.index).dropna(how='all')

                    # 8. 保存过滤后的新数据
                    if not indicator_result_df_filtered.empty:
                        # logger.debug(f"[{indicator_name}] 计算完成，准备保存 {len(indicator_result_df_filtered)} 条新数据 for {stock_code} {time_level_str}")
                        await save_func(stock_info, time_level_str, indicator_result_df_filtered)
                    # else:
                    #     logger.debug(f"[{indicator_name}] 计算完成，但所有时间点的数据都已存在，无需保存 for {stock_code} {time_level_str}")

                # else:
                #     logger.warning(f"[{indicator_name}] 计算结果为空或计算失败 for {stock_code} {time_level_str}")

            except Exception as e:
                logger.error(f"[{indicator_name}] 处理指标时发生严重错误 for {stock_code} {time_level_str}: {e}", exc_info=True)

        # logger.info(f"完成所有指标的检查、计算和保存 for {stock_code} {time_level_str}")

    # --- calculate_and_save_macd_indicators 方法 ---
    async def calculate_and_save_macd_indicators(self, stock_code: str, time_level: Union[TimeLevel, str]):
        """
        计算指定股票和时间级别的特定指标（如 MACD 相关），并保存到数据库。
        优化：如果最新 K 线数据对应的指标已存在，则跳过该指标的计算和保存。
              否则，计算指标并保存结果中数据库尚不存在的数据。
        """
        if ta is None:
            logger.error("pandas-ta 未加载，无法计算指标。")
            return

        stock_info = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock_info:
            logger.error(f"无法找到股票信息: {stock_code}，指标计算中止")
            return

        time_level_str = time_level.value if isinstance(time_level, TimeLevel) else str(time_level)
        # logger.info(f"开始检查并计算/保存 MACD 相关指标 for {stock_code} {time_level_str}")

        # 1. 获取基础 OHLCV 数据
        needed_bars = max(FIB_PERIODS) + 20
        ohlcv_df_raw = await self._get_ohlcv_data(stock_code, time_level, needed_bars)
        if ohlcv_df_raw is None or ohlcv_df_raw.empty:
            logger.error(f"无法获取用于计算指标的历史数据 for {stock_code} {time_level_str}")
            return

        # 2. 获取最新 K 线的时间戳
        latest_ohlcv_timestamp = ohlcv_df_raw.index[-1]
        # logger.debug(f"最新的 OHLCV 时间戳: {latest_ohlcv_timestamp} for {stock_code} {time_level_str}")

        timestamps_in_ohlcv = list(ohlcv_df_raw.index)

        # --- indicator_tasks 列表保持不变 (包含模型类) ---
        indicator_tasks: List[Tuple[callable, callable, Type[models.Model], Dict]] = [
            (self.calculate_macd_fib, self.indicator_dao.save_macd_fib, StockMACDFIB, {}),
            (self.calculate_rsi_fib, self.indicator_dao.save_rsi_fib, StockRsiFIB, {}),
            (self.calculate_kdj_fib, self.indicator_dao.save_kdj_fib, StockKDJFIB, {}),
            (self.calculate_boll, self.indicator_dao.save_boll, StockBOLLIndicator, {}),
            (self.calculate_cci_fib, self.indicator_dao.save_cci_fib, StockCciFIB, {}),
            (self.calculate_mfi_fib, self.indicator_dao.save_mfi_fib, StockMfiFIB, {}),
            (self.calculate_roc_fib, self.indicator_dao.save_roc_fib, StockRocFIB, {}),
            (self.calculate_dmi_fib, self.indicator_dao.save_dmi_fib, StockDmiFIB, {}),
            (self.calculate_sar, self.indicator_dao.save_sar, StockSar, {}),
            (self.calculate_amount_roc_fib, self.indicator_dao.save_amount_roc_fib, StockAmountRocFIB, {}),
            (self.calculate_cmf_fib, self.indicator_dao.save_cmf_fib, StockCmfFIB, {}),
            (self.calculate_obv, self.indicator_dao.save_obv, StockObvFIB, {}),
            (self.calculate_vwap, self.indicator_dao.save_vwap, StockVwap, {}),
            (self.calculate_ema_fib, self.indicator_dao.save_ema_fib, StockEmaFIB, {}),
        ]

        # --- 修改循环逻辑 (与 calculate_and_save_all_indicators 类似) ---
        for calc_func, save_func, model_class, params in indicator_tasks:
            indicator_name = calc_func.__name__.replace('calculate_', '').upper()

            try:
                # 3. 检查最新时间戳的指标数据是否已存在
                latest_data_exists = await self.indicator_dao.check_indicator_exists_at_timestamp(
                    stock_info, time_level_str, model_class, latest_ohlcv_timestamp
                )

                if latest_data_exists:
                    # logger.debug(f"[{indicator_name}] 最新时间戳 {latest_ohlcv_timestamp} 的数据已存在，跳过计算和保存 for {stock_code} {time_level_str}")
                    continue # 跳到下一个指标

                # logger.debug(f"[{indicator_name}] 最新时间戳数据不存在，开始计算 for {stock_code} {time_level_str}")

                # 4. 计算指标
                ohlcv_df_copy = ohlcv_df_raw.copy()
                indicator_result_df = calc_func(ohlcv_df_copy, **params)

                if indicator_result_df is not None and not indicator_result_df.empty:
                    # 确保结果索引是时区感知的
                    if not isinstance(indicator_result_df.index, pd.DatetimeIndex):
                         indicator_result_df.index = pd.to_datetime(indicator_result_df.index)
                    default_tz = timezone.get_default_timezone()
                    if indicator_result_df.index.tz is None:
                         indicator_result_df.index = indicator_result_df.index.tz_localize(default_tz)
                    elif indicator_result_df.index.tz != default_tz:
                         indicator_result_df.index = indicator_result_df.index.tz_convert(default_tz)

                    # 5. 查询数据库中已存在的时间戳 (用于过滤保存)
                    existing_timestamps_set = await self.indicator_dao.get_existing_timestamps_for_range(
                        stock_info, time_level_str, model_class, timestamps_in_ohlcv
                    )

                    # 6. 过滤掉计算结果中时间戳已存在于数据库的行
                    indicator_result_df_filtered = indicator_result_df[~indicator_result_df.index.isin(existing_timestamps_set)]

                    # 7. 对齐索引并去除全 NaN 行
                    indicator_result_df_filtered = indicator_result_df_filtered.reindex(ohlcv_df_raw.index).dropna(how='all')

                    # 8. 保存过滤后的新数据
                    if not indicator_result_df_filtered.empty:
                        # logger.debug(f"[{indicator_name}] 计算完成，准备保存 {len(indicator_result_df_filtered)} 条新数据 for {stock_code} {time_level_str}")
                        await save_func(stock_info, time_level_str, indicator_result_df_filtered)
                    # else:
                    #     logger.debug(f"[{indicator_name}] 计算完成，但所有时间点的数据都已存在，无需保存 for {stock_code} {time_level_str}")
                # else:
                #     logger.warning(f"[{indicator_name}] 计算结果为空或计算失败 for {stock_code} {time_level_str}")

            except Exception as e:
                logger.error(f"[{indicator_name}] 处理指标时发生严重错误 for {stock_code} {time_level_str}: {e}", exc_info=True)

    async def prepare_strategy_dataframe(self, stock_code: str, params_file: str) -> Optional[pd.DataFrame]:
        """
        根据策略 JSON 配置文件准备包含基础数据和所有计算指标的 DataFrame。
        Args:
            stock_code (str): 股票代码。
            params_file (str): 策略参数 JSON 文件的路径。
        Returns:
            Optional[pd.DataFrame]: 包含所有所需数据的 DataFrame，如果失败则返回 None。
                                     列名将包含时间级别后缀，例如 'RSI_12_15', 'close_60'。
        """
        if ta is None:
            logger.error(f"[{stock_code}] pandas_ta 未加载，无法准备策略数据。")
            return None
        # 1. 加载 JSON 参数
        try:
            if not os.path.exists(params_file):
                logger.error(f"[{stock_code}] 策略参数文件未找到: {params_file}")
                return None
            with open(params_file, 'r', encoding='utf-8') as f:
                params = json.load(f)
            logger.info(f"[{stock_code}] 从 {params_file} 加载策略参数成功。")
        except Exception as e:
            logger.error(f"[{stock_code}] 加载或解析参数文件 {params_file} 失败: {e}", exc_info=True)
            return None
        # 2. 识别需求: 时间级别和最大回看期
        all_time_levels = set()
        max_lookback = 0
        try:
            bs_params = params['base_scoring']
            vc_params = params['volume_confirmation']
            dd_params = params['divergence_detection']
            kpd_params = params['kline_pattern_detection']
            ta_params = params['trend_analysis']
            ia_params = params['indicator_analysis_params']

            all_time_levels.update(bs_params['timeframes'])
            if vc_params['enabled']: all_time_levels.add(vc_params['tf'])
            if dd_params['enabled']: all_time_levels.add(dd_params['tf'])
            if kpd_params['enabled']: all_time_levels.add(kpd_params['tf'])
            # 分析所需指标的时间框 (默认为 divergence tf 或 kline tf 或 volume tf)
            analysis_tf = dd_params.get('tf') if dd_params['enabled'] else \
                          kpd_params.get('tf') if kpd_params['enabled'] else \
                          vc_params.get('tf', bs_params['timeframes'][0]) # fallback
            all_time_levels.add(analysis_tf)
            # 计算最大回看期 (需要考虑所有指标参数)
            lookbacks = [
                bs_params.get('rsi_period', 0), bs_params.get('kdj_period_k', 0),
                bs_params.get('boll_period', 0), bs_params.get('macd_slow', 0) + bs_params.get('macd_signal', 0),
                bs_params.get('cci_period', 0), bs_params.get('mfi_period', 0),
                bs_params.get('roc_period', 0), bs_params.get('dmi_period', 0) * 3, # DMI/ADX 需要更多数据
                vc_params.get('amount_ma_period', 0), vc_params.get('cmf_period', 0), vc_params.get('obv_ma_period', 0),
                dd_params.get('lookback', 0) * 2, # find_peaks 可能需要 2*lookback
                max(ta_params.get('ema_periods', [0])), # 评分 EMA
                ia_params.get('stoch_k', 0) + ia_params.get('stoch_d', 0) + ia_params.get('stoch_smooth_k', 0),
                ia_params.get('volume_ma_period', 0),
                55 # SAR 默认lookback? 或者给个固定值
            ]
            max_lookback = max(lookbacks) + 50 # 加 50 bar 缓冲
            logger.info(f"[{stock_code}] 需要的时间级别: {all_time_levels}, 最大回看期: {max_lookback}")
        except KeyError as e:
            logger.error(f"[{stock_code}] 参数文件 {params_file} 缺少键: {e}", exc_info=True)
            return None
        # 3. 获取基础 OHLCV 数据
        ohlcv_tasks = {
            tf: self._get_ohlcv_data(stock_code, tf, max_lookback)
            for tf in all_time_levels
        }
        ohlcv_results = await asyncio.gather(*ohlcv_tasks.values())
        ohlcv_dfs = dict(zip(all_time_levels, ohlcv_results))

        # 检查是否有数据获取失败
        valid_ohlcv_dfs = {}
        for tf, df in ohlcv_dfs.items():
            if df is None or df.empty:
                 logger.warning(f"[{stock_code}] 无法获取时间级别 {tf} 的 OHLCV 数据。")
            else:
                 # 【修改】重命名基础列，将 turnover 重命名为 amount
                 rename_map = {}
                 for col in df.columns:
                      if col in ['open','high','low','close','volume']:
                           rename_map[col] = f"{col}_{tf}"
                      elif col == 'turnover': # 特别处理 turnover
                           rename_map[col] = f"amount_{tf}" # 重命名为 amount_{tf}
                      # else: 保留其他列名不变 (如 stock_code, time_level)
                 valid_ohlcv_dfs[tf] = df.rename(columns=rename_map)
        if not valid_ohlcv_dfs:
             logger.error(f"[{stock_code}] 无法获取任何有效的基础 OHLCV 数据。")
             return None
        # 4. 计算所有指标
        calculated_indicators = defaultdict(list) # {tf: [indicator_df1, indicator_df2, ...]}
        # --- Helper for safe calculation ---
        def _calculate_and_store(tf, indicator_name, calculation_func, *args, **kwargs):
            if tf in valid_ohlcv_dfs:
                base_df = ohlcv_dfs[tf] # 使用原始未重命名的 DF 进行计算
                if base_df is None or base_df.empty: return
                try:
                    # 确保基础数据是数值类型
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                         if col in base_df.columns:
                              base_df[col] = pd.to_numeric(base_df[col], errors='coerce')
                    base_df_clean = base_df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
                    if base_df_clean.empty:
                         logger.warning(f"[{stock_code}] 清理 NaN 后时间级别 {tf} 数据为空，无法计算 {indicator_name}")
                         return
                    result = calculation_func(base_df_clean, *args, **kwargs)
                    if result is not None and not result.empty:
                        # 【修改】统一为所有结果列添加时间后缀
                        # 检查是否已有后缀是为了避免旧方法加两次，现在统一加
                        result_renamed = result.rename(columns=lambda x: f"{x}_{tf}")
                        calculated_indicators[tf].append(result_renamed)
                except Exception as e:
                    logger.error(f"[{stock_code}] 计算指标 {indicator_name} (时间 {tf}) 时出错: {e}", exc_info=True)

        # --- Iterate through Base Scoring indicators ---
        for indi_key in bs_params.get('score_indicators', []):
            for tf in bs_params.get('timeframes', []):
                if indi_key == 'macd':
                    # 调用更新后的 calculate_macd
                    _calculate_and_store(tf, 'MACD', self.calculate_macd,
                                         period_fast=bs_params['macd_fast'],
                                         period_slow=bs_params['macd_slow'],
                                         signal_period=bs_params['macd_signal'])
                elif indi_key == 'rsi':
                    # 调用更新后的 calculate_rsi
                    _calculate_and_store(tf, 'RSI', self.calculate_rsi, period=bs_params['rsi_period'])
                elif indi_key == 'kdj':
                     # 调用更新后的 calculate_kdj
                     _calculate_and_store(tf, 'KDJ', self.calculate_kdj,
                                          period=bs_params['kdj_period_k'],
                                          signal_period=bs_params['kdj_period_d'],
                                          smooth_k_period=bs_params['kdj_period_j'])
                elif indi_key == 'boll':
                     # 调用 calculate_boll (它本来就接受参数)
                     _calculate_and_store(tf, 'BOLL', self.calculate_boll,
                                          period=bs_params['boll_period'],
                                          std_dev=bs_params['boll_std_dev'])
                elif indi_key == 'cci':
                    # 调用更新后的 calculate_cci
                    _calculate_and_store(tf, 'CCI', self.calculate_cci, period=bs_params['cci_period'])
                elif indi_key == 'mfi':
                    # 调用更新后的 calculate_mfi
                    _calculate_and_store(tf, 'MFI', self.calculate_mfi, period=bs_params['mfi_period'])
                elif indi_key == 'roc':
                    # 调用更新后的 calculate_roc
                    _calculate_and_store(tf, 'ROC', self.calculate_roc, period=bs_params['roc_period'])
                elif indi_key == 'dmi':
                    # 调用更新后的 calculate_dmi
                    _calculate_and_store(tf, 'DMI', self.calculate_dmi, period=bs_params['dmi_period'])
                elif indi_key == 'sar':
                    # 调用 calculate_sar (它本来就接受参数)
                     _calculate_and_store(tf, 'SAR', self.calculate_sar, af=bs_params['sar_step'], max_af=bs_params['sar_max'])

        # --- Iterate through Volume Confirmation indicators ---
        if vc_params['enabled']:
            tf = vc_params['tf']
            # Amount MA - 调用更新后的 calculate_amount_ma
            _calculate_and_store(tf, 'AMT_MA', self.calculate_amount_ma, period=vc_params['amount_ma_period'])
            # CMF - 调用更新后的 calculate_cmf
            _calculate_and_store(tf, 'CMF', self.calculate_cmf, period=vc_params['cmf_period'])
            # OBV - 调用 calculate_obv
            _calculate_and_store(tf, 'OBV', self.calculate_obv)
            # OBV MA (calculate separately after OBV)
            if tf in calculated_indicators:
                 obv_df = next((df for df in calculated_indicators[tf] if f'OBV_{tf}' in df.columns), None)
                 if obv_df is not None:
                      obv_series = obv_df[f'OBV_{tf}']
                      if not obv_series.isnull().all():
                           obv_ma_series = ta.sma(obv_series, length=vc_params['obv_ma_period'])
                           if obv_ma_series is not None:
                                obv_ma_df = obv_ma_series.to_frame(name=f'OBV_MA_{vc_params["obv_ma_period"]}_{tf}')
                                calculated_indicators[tf].append(obv_ma_df)
        # --- Indicators needed for analysis ---
        stoch_tf = analysis_tf
        # 调用更新后的 calculate_stoch
        _calculate_and_store(stoch_tf, 'STOCH', self.calculate_stoch,
                             k_period=ia_params['stoch_k'],
                             d_period=ia_params['stoch_d'],
                             smooth_k_period=ia_params['stoch_smooth_k'])
        # Volume MA (if not already calculated)
        vol_ma_tf = analysis_tf
        vol_ma_col_name = f'VOL_MA_{ia_params["volume_ma_period"]}_{vol_ma_tf}'
        vol_ma_exists = any(vol_ma_col_name in df.columns for df in calculated_indicators.get(vol_ma_tf, []))
        if not vol_ma_exists:
             # 直接用 ta.sma 计算并命名
             _calculate_and_store(vol_ma_tf, 'VOL_MA', lambda df: df.ta.sma(close='volume', length=ia_params['volume_ma_period']).to_frame(name=f'VOL_MA_{ia_params["volume_ma_period"]}'))
        # BBands (if not calculated)
        bb_tf = analysis_tf
        bb_col_name = f'BB_UPPER_{bb_tf}'
        bb_exists = any(bb_col_name in df.columns for df in calculated_indicators.get(bb_tf, []))
        if not bb_exists:
             _calculate_and_store(bb_tf, 'BOLL', self.calculate_boll, period=bs_params['boll_period'], std_dev=bs_params['boll_std_dev'])
        # VWAP (if needed)
        if params.get('t_plus_0_signals', {}).get('enabled'):
            vwap_tf = analysis_tf
            _calculate_and_store(vwap_tf, 'VWAP', self.calculate_vwap)
        # 5. 合并所有结果
        all_dataframes_to_merge = []
        # Add base OHLCV data first
        for tf, df in valid_ohlcv_dfs.items():
            # 【修改】确保只选择重命名后的基础列 (open_tf, ..., amount_tf)
            base_cols = [col for col in df.columns if col.startswith(('open_', 'high_', 'low_', 'close_', 'volume_', 'amount_'))]
            all_dataframes_to_merge.append(df[base_cols])
        # Add calculated indicators
        for tf, indi_list in calculated_indicators.items():
            all_dataframes_to_merge.extend(indi_list)
        if not all_dataframes_to_merge:
            logger.warning(f"[{stock_code}] 没有有效的 OHLCV 或计算指标可供合并。")
            return None
        # 合并，使用 outer join 保留所有时间戳
        try:
            # 使用 reduce 进行合并，更稳健处理多个 DataFrame
            final_df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), all_dataframes_to_merge)
        except Exception as merge_err:
            logger.error(f"[{stock_code}] 合并指标数据时出错: {merge_err}", exc_info=True)
            # 尝试打印每个 DataFrame 的信息以便调试
            # for i, df_merge in enumerate(all_dataframes_to_merge):
            #    logger.debug(f"DF {i} - Index type: {type(df_merge.index)}, Columns: {df_merge.columns.tolist()}")
            #    logger.debug(f"DF {i} Head:\n{df_merge.head()}")
            return None
        # --- 列名修正 (简化) ---
        # 由于 _calculate_and_store 现在统一添加后缀，并且 calculate_kdj 返回了修正的基础名，
        # 这里可能不再需要复杂的重命名。
        # 主要确认 KDJ 和 VWAP 的最终列名是否符合预期。
        # 预期 KDJ: K_9_3_15, D_9_3_15, J_9_3_15
        # 预期 VWAP: VWAP_15
        # 检查 final_df.columns 进行确认
        # logger.debug(f"[{stock_code}] 合并后 DataFrame 列名: {final_df.columns.tolist()}")
        # 【修改】添加 VWAP 列名小写化处理
        rename_final_map = {}
        # 查找所有 VWAP_{tf} 格式的列
        vwap_cols_to_rename = [col for col in final_df.columns if col.startswith('VWAP_')]
        for col in vwap_cols_to_rename:
             # 将 VWAP_tf 重命名为 vwap_tf
             tf_suffix = col.split('_')[-1] # 提取时间框后缀
             rename_final_map[col] = f'vwap_{tf_suffix}' # 生成小写目标列名
        # 如果仍然发现列名不匹配，可以在这里添加必要的 rename 操作，
        # 例如:
        # rename_final_map = {}
        # if f'SomeWrongName_{tf}' in final_df.columns:
        #     rename_final_map[f'SomeWrongName_{tf}'] = f'CorrectName_{tf}'
        # final_df.rename(columns=rename_final_map, inplace=True, errors='ignore')
        # 应用所有重命名
        if rename_final_map:
            final_df.rename(columns=rename_final_map, inplace=True, errors='ignore')
            # logger.debug(f"[{stock_code}] 应用的列名重命名映射: {rename_final_map}")
        # 确保索引是排序的
        final_df.sort_index(inplace=True)
        # logger.info(f"[{stock_code}] 策略数据准备完成。最终 DataFrame Shape: {final_df.shape}, Columns: {final_df.columns.tolist()}")
        # logger.info(f"[{stock_code}] 策略数据准备完成。最终 DataFrame last one:\n{final_df.tail().get('J_7_3_D')}")
        # logger.debug(f"[{stock_code}] Final DataFrame Head:\n{final_df.head()}")
        # logger.debug(f"[{stock_code}] Final DataFrame Tail:\n{final_df.tail()}")
        return final_df
    
    # --- 新增方法：准备基础 OHLCV 数据 ---
    async def prepare_strategy_basic_dataframe(self, stock_code: str, timeframes: List[str],
                                                 limit_per_tf: int = 1200) -> Optional[pd.DataFrame]:
        """
        准备策略所需的基础 OHLCV DataFrame (简化版)。
        仅获取并合并指定时间周期的 OHLCV 和 Turnover 数据。

        Args:
            stock_code (str): 股票代码。
            timeframes (List[str]): 策略需要的时间周期列表 (e.g., ['5', '15', 'Day', 'Week'])。
                                     数据库存储格式: ['5','15','30','60','Day','Week','Month','Year']
            limit_per_tf (int): 为每个时间周期获取的最新记录数。

        Returns:
            Optional[pd.DataFrame]: 合并后的 DataFrame，索引为 trade_time (升序, timezone-aware)，
                                     列名为带后缀的 OHLCV 和 amount (来自 turnover)。
                                     如果数据准备失败，则返回 None。
        """
        logger.info(f"[{stock_code}] 开始准备基础策略 DataFrame for timeframes: {timeframes}")

        # --- 1. 处理和映射时间周期 ---
        tf_map = {'Day': 'D', 'Week': 'W', 'Month': 'M'}
        valid_timeframes_dao = [] # 用于 DAO 查询的周期
        valid_timeframes_orig = [] # 用于列名后缀的原始周期
        for tf in timeframes:
            if tf.isdigit(): # 数字周期直接使用
                valid_timeframes_dao.append(tf)
                valid_timeframes_orig.append(tf)
            elif tf in tf_map: # 映射 Day, Week, Month
                valid_timeframes_dao.append(tf_map[tf])
                valid_timeframes_orig.append(tf) # 后缀使用原始名称
            else:
                logger.warning(f"[{stock_code}] 不支持或无法映射的时间周期: '{tf}'，将被忽略。")

        if not valid_timeframes_dao:
            logger.error(f"[{stock_code}] 没有有效的 timeframes 可供查询。")
            return None

        # --- 2. 创建并发任务列表 ---
        tasks = []
        task_descriptions = {} # 用于追踪任务

        for i, tf_dao in enumerate(valid_timeframes_dao):
            tf_orig = valid_timeframes_orig[i] # 获取对应的原始名称用于后缀
            ohlcv_task = self.indicator_dao.get_history_ohlcv_df(stock_code, tf_dao, limit=limit_per_tf)
            tasks.append(ohlcv_task)
            # 描述信息包含 DAO 使用的周期和原始周期（用于后缀）
            task_descriptions[len(tasks)-1] = {'type': 'ohlcv', 'tf_dao': tf_dao, 'tf_orig': tf_orig}

        # --- 3. 并发执行所有 DAO 查询 ---
        logger.debug(f"[{stock_code}] 开始并发执行 {len(tasks)} 个 OHLCV 数据获取任务...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        logger.debug(f"[{stock_code}] OHLCV 数据获取任务执行完毕.")

        # --- 4. 处理结果并重命名列 ---
        dfs_to_merge: List[pd.DataFrame] = []
        fetched_data_summary = {}

        for i, result in enumerate(results):
            desc = task_descriptions[i]
            tf_orig = desc['tf_orig'] # 使用原始周期名作为后缀
            tf_dao = desc['tf_dao']   # 用于日志或调试
            data_key = f"ohlcv_{tf_orig}" # 使用原始名构建 key

            if isinstance(result, Exception):
                logger.warning(f"[{stock_code}] 获取 {data_key} (DAO: {tf_dao}) 数据时出错: {result}", exc_info=False)
                fetched_data_summary[data_key] = "Error"
                continue
            if result is None or result.empty:
                # logger.warning(f"[{stock_code}] 未获取到有效的 {data_key} (DAO: {tf_dao}) 数据")
                fetched_data_summary[data_key] = "Empty/None"
                continue

            df = result.copy() # 操作副本

            # --- 定义重命名映射 ---
            # DAO 返回的列名是 'open', 'high', 'low', 'close', 'volume', 'turnover' (假设)
            # 我们需要将它们重命名为带原始后缀的列名
            rename_map = {
                'open': f'open_{tf_orig}',
                'high': f'high_{tf_orig}',
                'low': f'low_{tf_orig}',
                'close': f'close_{tf_orig}',
                'volume': f'volume_{tf_orig}',
                'turnover': f'amount_{tf_orig}' # 将 turnover 重命名为 amount 并加后缀
            }

            # --- 执行重命名并检查 ---
            try:
                # 检查原始列是否存在
                missing_original_cols = [orig_col for orig_col in rename_map.keys() if orig_col not in df.columns]
                if missing_original_cols:
                    logger.warning(f"[{stock_code}] 获取的 {data_key} 数据缺少原始列: {missing_original_cols}. 可用列: {df.columns.tolist()}")
                    # 移除缺失的映射，继续处理存在的列
                    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
                    if not rename_map: # 如果所有需要的列都缺失
                        fetched_data_summary[data_key] = "Missing All Original Cols"
                        continue # 跳过这个 DataFrame

                df.rename(columns=rename_map, inplace=True)
                renamed_cols = list(rename_map.values())

                # 只保留重命名后的列
                df_to_add = df[renamed_cols]

                # --- 确保索引是 DatetimeIndex 且时区感知 ---
                if not isinstance(df_to_add.index, pd.DatetimeIndex):
                    try:
                        df_to_add.index = pd.to_datetime(df_to_add.index)
                    except Exception as e_idx:
                        logger.warning(f"[{stock_code}] 无法将 {data_key} 的索引转换为 DatetimeIndex: {e_idx}")
                        fetched_data_summary[data_key] = "Index Error"
                        continue # 跳过这个 DataFrame

                # 确保时区（与 Django 默认时区一致）
                target_tz_final = timezone.get_default_timezone()
                try:
                    if df_to_add.index.tz is None:
                        df_to_add.index = df_to_add.index.tz_localize('UTC').tz_convert(target_tz_final)
                    elif df_to_add.index.tz != target_tz_final:
                        df_to_add.index = df_to_add.index.tz_convert(target_tz_final)
                except Exception as e_tz:
                    logger.warning(f"[{stock_code}] 处理 {data_key} 索引时区时出错: {e_tz}")
                    fetched_data_summary[data_key] = "Timezone Error"
                    continue # 跳过

                dfs_to_merge.append(df_to_add)
                fetched_data_summary[data_key] = "Success"

            except Exception as e_process:
                logger.error(f"[{stock_code}] 处理或重命名 {data_key} 列时出错: {e_process}", exc_info=True)
                fetched_data_summary[data_key] = "Processing Exception"

        # logger.info(f"[{stock_code}] 基础数据获取与初步处理概要: {fetched_data_summary}")

        # --- 5. 合并 DataFrame ---
        if not dfs_to_merge:
            logger.error(f"[{stock_code}] 没有成功获取或处理任何有效的 OHLCV 数据用于合并。")
            return None

        try:
            logger.debug(f"[{stock_code}] 开始合并 {len(dfs_to_merge)} 个基础 OHLCV DataFrame...")

            # 使用 concat 进行合并，基于索引对齐
            merged_df = pd.concat(dfs_to_merge, axis=1, join='outer') # outer join 保留所有时间点
            merged_df.sort_index(ascending=True, inplace=True)

            # --- 6. 填充 NaN 值 ---
            # 向前填充通常是合理的策略，表示沿用上一时间点的值
            # 对于不同频率的数据合并，这可能导致低频数据在高频时间点重复
            # 但对于基础数据，这通常是期望的行为（例如，日线的开盘价在当天所有5分钟线上都一样）
            merged_df.ffill(inplace=True)

            # （可选）清理：如果需要，可以在这里加入基于某个关键列（如 close_5）删除头部 NaN 的逻辑
            # key_col = f'close_{valid_timeframes_orig[0]}' # 例如用第一个周期的 close
            # if key_col in merged_df.columns:
            #     merged_df.dropna(subset=[key_col], inplace=True)

            # 最后检查是否为空
            if merged_df.empty:
                logger.error(f"[{stock_code}] 合并并清理基础 OHLCV 数据后 DataFrame 为空。")
                return None

            logger.info(f"[{stock_code}] 基础策略 DataFrame 准备完成，最终形状: {merged_df.shape}")
            # logger.debug(f"[{stock_code}] 最终列名: {merged_df.columns.tolist()}")
            return merged_df

        except Exception as e_merge:
            logger.error(f"[{stock_code}] 合并或后处理基础 OHLCV DataFrame 时出错: {e_merge}", exc_info=True)
            return None












