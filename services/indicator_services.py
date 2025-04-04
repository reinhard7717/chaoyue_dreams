# indicator_services.py (使用 DAO 重构版本 - v4，调用新 DAO 保存)

import pandas as pd
import numpy as np
import asyncio
import logging
from typing import Type # 确保导入了 Type
from django.db import models # <--- 添加这一行导入
from django.utils import timezone
from datetime import timedelta, datetime, date
from typing import List, Optional, Dict, Type
from asgiref.sync import sync_to_async

# 模型和 DAO 导入
from stock_models.stock_basic import StockInfo
from stock_models.stock_indicators import (
    StockAmountMaIndicator, StockAmountRocIndicator, StockAtrIndicator, StockCciIndicator, StockCmfIndicator, StockDmiIndicator, StockEmaIndicator, StockMfiIndicator, StockMomIndicator, StockObvIndicator, StockRocIndicator, StockRsiIndicator, StockSarIndicator, StockTimeTrade, StockKDJIndicator, StockMACDIndicator, StockMAIndicator, StockBOLLIndicator, StockVrocIndicator, StockVwapIndicator, StockWrIndicator
)
from dao_manager.daos.stock_basic_dao import StockBasicDAO
from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO, TIME_LEVELS
# 导入新的 DAO
from dao_manager.daos.self_built_indicators_dao import SelfBuiltIndicatorsDao

# 导入 finta
from finta import TA

# 定义常量
FIB_PERIODS = (5, 8, 13, 21, 34, 55)
INDICATOR_LOOKBACK_PERIOD = max(FIB_PERIODS) * 3 if FIB_PERIODS else 200

logger = logging.getLogger("services")

class BaseIndicatorService:
    """
    指标计算服务基类 (调用 DAO 获取数据和存储)
    """
    indicator_model: Type[models.Model] = None
    indicator_name: str = "Base"
    finta_required_columns: List[str] = ['open', 'high', 'low', 'close', 'volume']

    def __init__(self):
        self.stock_basic_dao = StockBasicDAO()
        self.stock_indicators_dao = StockIndicatorsDAO()
        self.self_built_dao = SelfBuiltIndicatorsDao() # 实例化新的 DAO

    async def _fetch_kline_data_from_dao(self, stock_code: str, kline_period: str) -> pd.DataFrame | None:
        """
        通过 StockIndicatorsDAO 获取历史 K 线数据并转换为 DataFrame。
        (此方法逻辑与上一个版本相同)
        """
        try:
            klines_list: List[Dict] = await self.stock_indicators_dao.get_history_time_trades_by_limit(
                stock_code=stock_code,
                time_level=kline_period,
                limit=INDICATOR_LOOKBACK_PERIOD
            )
            if not klines_list or len(klines_list) < min(FIB_PERIODS if FIB_PERIODS else [1]):
                logger.warning(f"警告: 从 DAO 获取的股票 {stock_code} 周期 {kline_period} 的K线数据不足 ({len(klines_list) if klines_list else 0}条)。")
                return None

            df = pd.DataFrame(klines_list)
            column_mapping = {
                'trade_time': 'timestamp', 'open_price': 'open', 'high_price': 'high',
                'low_price': 'low', 'close_price': 'close', 'volume': 'volume',
                'turnover': 'amount' # 注意这里是 turnover 映射到 amount
            }
            rename_dict = {}
            existing_cols = []
            for dao_key, finta_key in column_mapping.items():
                if dao_key in df.columns:
                    rename_dict[dao_key] = finta_key
                    existing_cols.append(dao_key)
            df = df[existing_cols].rename(columns=rename_dict)

            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp'])
            else: return None

            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
            for col in numeric_cols:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    if col in self.finta_required_columns: return None # 缺少核心列

            df = df.sort_values(by='timestamp').reset_index(drop=True)

            if not all(col in df.columns for col in self.finta_required_columns):
                 missing = [col for col in self.finta_required_columns if col not in df.columns]
                 logger.error(f"错误: DataFrame 缺少计算 {self.indicator_name} 所需的 finta 列: {missing}")
                 return None
            if df[self.finta_required_columns].isnull().any().any():
                 logger.warning(f"警告: 股票 {stock_code} 周期 {kline_period} 的K线数据包含 NaN 值。")
                 # 可以考虑填充 df[self.finta_required_columns] = df[self.finta_required_columns].ffill()

            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"从 DAO 获取并处理 K线 DataFrame 时出错 ({stock_code}, {kline_period}): {e}")
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """
        计算指标的核心方法。子类需要实现此方法。
        """
        raise NotImplementedError("子类必须实现 calculate_indicators 方法")

    async def calculate_and_save_stock(self, stock_code: str, kline_period: str):
        """
        计算单个股票、单个周期的指标，并调用 DAO 保存。
        """
        # 确保子类设置了 indicator_model
        if self.indicator_model is None:
             logger.error(f"错误: 服务 {self.__class__.__name__} 未设置 indicator_model。")
             return

        logger.info(f"开始计算指标 {self.indicator_name} for {stock_code} ({kline_period})...")
        df = await self._fetch_kline_data_from_dao(stock_code, kline_period)
        if df is None or df.empty:
            logger.warning(f"无法获取 {stock_code} ({kline_period}) 的 K 线数据，跳过计算。")
            return

        indicator_df = self.calculate_indicators(df)
        if indicator_df is None or indicator_df.empty:
             logger.warning(f"无法计算 {stock_code} ({kline_period}) 的 {self.indicator_name} 指标，跳过保存。")
             return

        # 调用 SelfBuiltIndicatorsDao 的保存方法
        save_result = await self.self_built_dao.save_indicator_data(
            stock_code=stock_code,
            kline_period=kline_period,
            indicator_model_class=self.indicator_model, # 传递模型类
            indicator_df=indicator_df
        )
        logger.info(f"完成指标 {self.indicator_name} for {stock_code} ({kline_period}) 的计算和保存尝试，结果: {save_result}")

    async def calculate_and_save_all(self, kline_period: str):
        """
        计算并保存所有股票指定周期的指标。
        """
        logger.info(f"开始计算所有股票的 {self.indicator_name} 指标 ({kline_period})...")
        all_stocks_info = await self.stock_basic_dao.get_stock_list()
        if not all_stocks_info:
            logger.error("错误：无法获取股票列表，无法执行批量计算。")
            return

        total_stocks = len(all_stocks_info)
        processed_count = 0
        start_time = timezone.now()

        # 可以考虑使用 asyncio.gather 来并发执行部分任务，但要注意数据库连接和 API 限制
        tasks = []
        for stock_info in all_stocks_info:
            # 创建计算和保存的任务
            task = self.calculate_and_save_stock(stock_info.code, kline_period)
            tasks.append(task)
            # 可以分批执行 gather，避免一次性创建过多任务
            if len(tasks) >= 50: # 例如每 50 个并发一次
                 await asyncio.gather(*tasks)
                 tasks = []
                 processed_count += 50
                 elapsed_time = timezone.now() - start_time
                 logger.info(f"进度 ({self.indicator_name}, {kline_period}): {processed_count}/{total_stocks} ({elapsed_time})")

        # 处理剩余的任务
        if tasks:
            await asyncio.gather(*tasks)
            processed_count += len(tasks)

        end_time = timezone.now()
        logger.info(f"完成所有股票 {self.indicator_name} ({kline_period}) 的计算和保存尝试，总耗时: {end_time - start_time}")

    async def get_indicators(self, stock_code: str, kline_period: str, start_date=None, end_date=None) -> list[dict] | None:
        """
        获取指定范围的指标数据 (通过 DAO)。
        """
        if self.indicator_model is None:
             logger.error(f"错误: 服务 {self.__class__.__name__} 未设置 indicator_model。")
             return None

        return await self.self_built_dao.get_indicator_data(
            stock_code=stock_code,
            kline_period=kline_period,
            indicator_model_class=self.indicator_model,
            start_date=start_date,
            end_date=end_date
        )


# --- 高优先级指标服务 ---
class HighPriorityIndicatorService(BaseIndicatorService):
    """高优先级指标计算服务: EMA, RSI, ATR, OBV"""

    # --- EMA ---
    async def calculate_and_save_ema_stock(self, stock_code: str, kline_period: str):
        self.indicator_model = StockEmaIndicator
        self.indicator_name = "EMA"
        self.finta_required_columns = ['close'] # EMA 只需要收盘价
        await self.calculate_and_save_stock(stock_code, kline_period)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """计算 EMA 指标"""
        if df is None or df.empty or 'close' not in df.columns: return None
        try:
            results = {}
            for period in FIB_PERIODS:
                if len(df) >= period: results[f'ema{period}'] = TA.EMA(df, period)
                else: results[f'ema{period}'] = np.nan
            return pd.DataFrame(results, index=df.index)
        except Exception as e: logger.error(f"计算 EMA 时出错: {e}"); return None

    async def get_ema_indicators(self, stock_code: str, kline_period: str, start_date=None, end_date=None):
        self.indicator_model = StockEmaIndicator
        self.indicator_name = "EMA"
        return await self.get_indicators(stock_code, kline_period, start_date, end_date)

    # --- RSI ---
    async def calculate_and_save_rsi_stock(self, stock_code: str, kline_period: str):
        self.indicator_model = StockRsiIndicator
        self.indicator_name = "RSI"
        self.finta_required_columns = ['close']
        await self.calculate_and_save_stock(stock_code, kline_period)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """计算 RSI 指标"""
        if df is None or df.empty or 'close' not in df.columns: return None
        try:
            results = {}
            for period in FIB_PERIODS:
                 if len(df) >= period + 1: results[f'rsi{period}'] = TA.RSI(df, period)
                 else: results[f'rsi{period}'] = np.nan
            return pd.DataFrame(results, index=df.index)
        except Exception as e: logger.error(f"计算 RSI 时出错: {e}"); return None

    async def get_rsi_indicators(self, stock_code: str, kline_period: str, start_date=None, end_date=None):
        self.indicator_model = StockRsiIndicator
        self.indicator_name = "RSI"
        return await self.get_indicators(stock_code, kline_period, start_date, end_date)

    # --- ATR ---
    async def calculate_and_save_atr_stock(self, stock_code: str, kline_period: str):
        self.indicator_model = StockAtrIndicator
        self.indicator_name = "ATR"
        self.finta_required_columns = ['high', 'low', 'close']
        await self.calculate_and_save_stock(stock_code, kline_period)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """计算 ATR 指标"""
        required = ['high', 'low', 'close']
        if df is None or df.empty or not all(c in df.columns for c in required): return None
        try:
            results = {}
            for period in FIB_PERIODS:
                 if len(df) >= period: results[f'atr{period}'] = TA.ATR(df, period)
                 else: results[f'atr{period}'] = np.nan
            return pd.DataFrame(results, index=df.index)
        except Exception as e: logger.error(f"计算 ATR 时出错: {e}"); return None

    async def get_atr_indicators(self, stock_code: str, kline_period: str, start_date=None, end_date=None):
        self.indicator_model = StockAtrIndicator
        self.indicator_name = "ATR"
        return await self.get_indicators(stock_code, kline_period, start_date, end_date)

    # --- OBV ---
    async def calculate_and_save_obv_stock(self, stock_code: str, kline_period: str):
        self.indicator_model = StockObvIndicator
        self.indicator_name = "OBV"
        self.finta_required_columns = ['close', 'volume']
        await self.calculate_and_save_stock(stock_code, kline_period)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """计算 OBV 指标"""
        required = ['close', 'volume']
        if df is None or df.empty or not all(c in df.columns for c in required): return None
        try:
            results = {'obv': TA.OBV(df)}
            return pd.DataFrame(results, index=df.index)
        except Exception as e: logger.error(f"计算 OBV 时出错: {e}"); return None

    async def get_obv_indicators(self, stock_code: str, kline_period: str, start_date=None, end_date=None):
        self.indicator_model = StockObvIndicator
        self.indicator_name = "OBV"
        return await self.get_indicators(stock_code, kline_period, start_date, end_date)


# --- 中优先级指标服务 ---
class MediumPriorityIndicatorService(BaseIndicatorService):
    """中优先级指标计算服务: DMI, CCI, WR, CMF, MFI"""

    # --- DMI ---
    async def calculate_and_save_dmi_stock(self, stock_code: str, kline_period: str):
        self.indicator_model = StockDmiIndicator
        self.indicator_name = "DMI"
        self.finta_required_columns = ['high', 'low', 'close']
        await self.calculate_and_save_stock(stock_code, kline_period)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """计算 DMI, ADX, ADXR 指标"""
        required = ['high', 'low', 'close']
        if df is None or df.empty or not all(c in df.columns for c in required): return None
        try:
            results = {}
            dmi_period = 13 # 使用斐波那契数 13
            if len(df) >= dmi_period:
                dmi_df = TA.DMI(df, period=dmi_period)
                results[f'plus_di{dmi_period}'] = dmi_df['DI+']
                results[f'minus_di{dmi_period}'] = dmi_df['DI-']
                # 检查并计算 ADX
                if hasattr(TA, 'ADX'):
                    # ADX 计算通常需要比 DMI 更多的周期，确保数据足够
                    adx_required_len = dmi_period * 2 # 经验值，可能需要调整
                    if len(df) >= adx_required_len:
                        results[f'adx{dmi_period}'] = TA.ADX(df, period=dmi_period)
                    else:
                        results[f'adx{dmi_period}'] = np.nan
                else: results[f'adx{dmi_period}'] = np.nan
                # ADXR 计算 (通常是 ADX 的移动平均或简单平均)
                # finta 可能不直接提供 ADXR，需要手动计算
                if f'adx{dmi_period}' in results and not results[f'adx{dmi_period}'].isnull().all():
                     # 简单实现：(ADX + n日前ADX) / 2，这里用 shift
                     adxr_shift = dmi_period # ADXR 通常参考 ADX 的周期
                     if len(df) >= adx_required_len + adxr_shift:
                         results[f'adxr{dmi_period}'] = (results[f'adx{dmi_period}'] + results[f'adx{dmi_period}'].shift(adxr_shift)) / 2
                     else:
                         results[f'adxr{dmi_period}'] = np.nan
                else: results[f'adxr{dmi_period}'] = np.nan
            else:
                results[f'plus_di{dmi_period}'] = np.nan
                results[f'minus_di{dmi_period}'] = np.nan
                results[f'adx{dmi_period}'] = np.nan
                results[f'adxr{dmi_period}'] = np.nan
            return pd.DataFrame(results, index=df.index)
        except Exception as e: logger.error(f"计算 DMI/ADX 时出错: {e}"); return None

    async def get_dmi_indicators(self, stock_code: str, kline_period: str, start_date=None, end_date=None):
        self.indicator_model = StockDmiIndicator
        self.indicator_name = "DMI"
        return await self.get_indicators(stock_code, kline_period, start_date, end_date)

    # --- CCI ---
    async def calculate_and_save_cci_stock(self, stock_code: str, kline_period: str):
        self.indicator_model = StockCciIndicator
        self.indicator_name = "CCI"
        self.finta_required_columns = ['high', 'low', 'close']
        await self.calculate_and_save_stock(stock_code, kline_period)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """计算 CCI 指标"""
        required = ['high', 'low', 'close']
        if df is None or df.empty or not all(c in df.columns for c in required): return None
        try:
            results = {}
            for period in FIB_PERIODS:
                if len(df) >= period: results[f'cci{period}'] = TA.CCI(df, period)
                else: results[f'cci{period}'] = np.nan
            return pd.DataFrame(results, index=df.index)
        except Exception as e: logger.error(f"计算 CCI 时出错: {e}"); return None

    async def get_cci_indicators(self, stock_code: str, kline_period: str, start_date=None, end_date=None):
        self.indicator_model = StockCciIndicator
        self.indicator_name = "CCI"
        return await self.get_indicators(stock_code, kline_period, start_date, end_date)

    # --- WR ---
    async def calculate_and_save_wr_stock(self, stock_code: str, kline_period: str):
        self.indicator_model = StockWrIndicator
        self.indicator_name = "WR"
        self.finta_required_columns = ['high', 'low', 'close']
        await self.calculate_and_save_stock(stock_code, kline_period)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """计算 WR 指标"""
        required = ['high', 'low', 'close']
        if df is None or df.empty or not all(c in df.columns for c in required): return None
        try:
            results = {}
            for period in FIB_PERIODS:
                 if len(df) >= period: results[f'wr{period}'] = TA.WILLIAMS(df, period) # finta 函数名是 WILLIAMS
                 else: results[f'wr{period}'] = np.nan
            return pd.DataFrame(results, index=df.index)
        except Exception as e: logger.error(f"计算 WR 时出错: {e}"); return None

    async def get_wr_indicators(self, stock_code: str, kline_period: str, start_date=None, end_date=None):
        self.indicator_model = StockWrIndicator
        self.indicator_name = "WR"
        return await self.get_indicators(stock_code, kline_period, start_date, end_date)

    # --- CMF ---
    async def calculate_and_save_cmf_stock(self, stock_code: str, kline_period: str):
        self.indicator_model = StockCmfIndicator
        self.indicator_name = "CMF"
        self.finta_required_columns = ['high', 'low', 'close', 'volume']
        await self.calculate_and_save_stock(stock_code, kline_period)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """计算 CMF 指标"""
        required = ['high', 'low', 'close', 'volume']
        if df is None or df.empty or not all(c in df.columns for c in required): return None
        try:
            results = {}
            cmf_period = 21 # 使用斐波那契数 21
            if len(df) >= cmf_period: results[f'cmf{cmf_period}'] = TA.CMF(df, period=cmf_period)
            else: results[f'cmf{cmf_period}'] = np.nan
            # 可以添加其他斐波那契周期的 CMF
            # period = 13
            # if len(df) >= period: results[f'cmf{period}'] = TA.CMF(df, period=period)
            # else: results[f'cmf{period}'] = np.nan
            return pd.DataFrame(results, index=df.index)
        except Exception as e: logger.error(f"计算 CMF 时出错: {e}"); return None

    async def get_cmf_indicators(self, stock_code: str, kline_period: str, start_date=None, end_date=None):
        self.indicator_model = StockCmfIndicator
        self.indicator_name = "CMF"
        return await self.get_indicators(stock_code, kline_period, start_date, end_date)

    # --- MFI ---
    async def calculate_and_save_mfi_stock(self, stock_code: str, kline_period: str):
        self.indicator_model = StockMfiIndicator
        self.indicator_name = "MFI"
        self.finta_required_columns = ['high', 'low', 'close', 'volume']
        await self.calculate_and_save_stock(stock_code, kline_period)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """计算 MFI 指标"""
        required = ['high', 'low', 'close', 'volume']
        if df is None or df.empty or not all(c in df.columns for c in required): return None
        try:
            results = {}
            mfi_period = 13 # 使用斐波那契数 13
            if len(df) >= mfi_period: results[f'mfi{mfi_period}'] = TA.MFI(df, period=mfi_period)
            else: results[f'mfi{mfi_period}'] = np.nan
            # 可以添加其他斐波那契周期的 MFI
            return pd.DataFrame(results, index=df.index)
        except Exception as e: logger.error(f"计算 MFI 时出错: {e}"); return None

    async def get_mfi_indicators(self, stock_code: str, kline_period: str, start_date=None, end_date=None):
        self.indicator_model = StockMfiIndicator
        self.indicator_name = "MFI"
        return await self.get_indicators(stock_code, kline_period, start_date, end_date)


# --- 低优先级指标服务 ---
class LowPriorityIndicatorService(BaseIndicatorService):
    """低优先级指标计算服务: SAR, ROC, MOM, VROC, AmountMA, AmountROC, VWAP"""
    # 注意：Ichimoku 暂时移除，因其多输出和 finta 实现细节待确认

    # --- SAR ---
    async def calculate_and_save_sar_stock(self, stock_code: str, kline_period: str):
        self.indicator_model = StockSarIndicator
        self.indicator_name = "SAR"
        self.finta_required_columns = ['high', 'low'] # SAR 主要需要高低价
        await self.calculate_and_save_stock(stock_code, kline_period)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """计算 SAR 指标"""
        required = ['high', 'low']
        if df is None or df.empty or not all(c in df.columns for c in required): return None
        try:
            # finta 的 SAR 可能需要调整参数，这里使用默认
            results = {'sar': TA.SAR(df)}
            return pd.DataFrame(results, index=df.index)
        except Exception as e: logger.error(f"计算 SAR 时出错: {e}"); return None

    async def get_sar_indicators(self, stock_code: str, kline_period: str, start_date=None, end_date=None):
        self.indicator_model = StockSarIndicator
        self.indicator_name = "SAR"
        return await self.get_indicators(stock_code, kline_period, start_date, end_date)

    # --- ROC ---
    async def calculate_and_save_roc_stock(self, stock_code: str, kline_period: str):
        self.indicator_model = StockRocIndicator
        self.indicator_name = "ROC"
        self.finta_required_columns = ['close']
        await self.calculate_and_save_stock(stock_code, kline_period)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """计算 ROC 指标"""
        if df is None or df.empty or 'close' not in df.columns: return None
        try:
            results = {}
            for period in FIB_PERIODS:
                if len(df) >= period + 1: results[f'roc{period}'] = TA.ROC(df, period=period)
                else: results[f'roc{period}'] = np.nan
            return pd.DataFrame(results, index=df.index)
        except Exception as e: logger.error(f"计算 ROC 时出错: {e}"); return None

    async def get_roc_indicators(self, stock_code: str, kline_period: str, start_date=None, end_date=None):
        self.indicator_model = StockRocIndicator
        self.indicator_name = "ROC"
        return await self.get_indicators(stock_code, kline_period, start_date, end_date)

    # --- MOM ---
    async def calculate_and_save_mom_stock(self, stock_code: str, kline_period: str):
        self.indicator_model = StockMomIndicator
        self.indicator_name = "MOM"
        self.finta_required_columns = ['close']
        await self.calculate_and_save_stock(stock_code, kline_period)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """计算 MOM 指标"""
        if df is None or df.empty or 'close' not in df.columns: return None
        try:
            results = {}
            for period in FIB_PERIODS:
                 if len(df) >= period: results[f'mom{period}'] = TA.MOM(df, period=period)
                 else: results[f'mom{period}'] = np.nan
            return pd.DataFrame(results, index=df.index)
        except Exception as e: logger.error(f"计算 MOM 时出错: {e}"); return None

    async def get_mom_indicators(self, stock_code: str, kline_period: str, start_date=None, end_date=None):
        self.indicator_model = StockMomIndicator
        self.indicator_name = "MOM"
        return await self.get_indicators(stock_code, kline_period, start_date, end_date)

    # --- VROC ---
    async def calculate_and_save_vroc_stock(self, stock_code: str, kline_period: str):
        self.indicator_model = StockVrocIndicator
        self.indicator_name = "VROC"
        self.finta_required_columns = ['volume'] # VROC 只需要成交量
        await self.calculate_and_save_stock(stock_code, kline_period)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """计算 VROC 指标"""
        if df is None or df.empty or 'volume' not in df.columns: return None
        try:
            results = {}
            # 检查 finta 是否有 VROC，如果没有，使用 ROC 计算 volume
            if hasattr(TA, 'VROC'):
                for period in FIB_PERIODS:
                    if len(df) >= period + 1: results[f'vroc{period}'] = TA.VROC(df, period=period)
                    else: results[f'vroc{period}'] = np.nan
            else:
                # 手动用 ROC 计算 volume 的变化率
                volume_df = df[['volume']].copy() # 创建只包含 volume 的 DataFrame
                volume_df.rename(columns={'volume': 'close'}, inplace=True) # 欺骗 ROC 函数
                for period in FIB_PERIODS:
                    if len(df) >= period + 1: results[f'vroc{period}'] = TA.ROC(volume_df, period=period)
                    else: results[f'vroc{period}'] = np.nan
            return pd.DataFrame(results, index=df.index)
        except Exception as e: logger.error(f"计算 VROC 时出错: {e}"); return None

    async def get_vroc_indicators(self, stock_code: str, kline_period: str, start_date=None, end_date=None):
        self.indicator_model = StockVrocIndicator
        self.indicator_name = "VROC"
        return await self.get_indicators(stock_code, kline_period, start_date, end_date)

    # --- Amount MA ---
    async def calculate_and_save_amount_ma_stock(self, stock_code: str, kline_period: str):
        self.indicator_model = StockAmountMaIndicator
        self.indicator_name = "AmountMA"
        self.finta_required_columns = ['amount'] # 需要成交额
        await self.calculate_and_save_stock(stock_code, kline_period)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """计算 Amount MA 指标 (使用 SMA)"""
        if df is None or df.empty or 'amount' not in df.columns: return None
        try:
            results = {}
            # 使用 SMA 计算成交额的移动平均
            amount_df = df[['amount']].copy()
            amount_df.rename(columns={'amount': 'close'}, inplace=True) # 欺骗 SMA 函数
            for period in FIB_PERIODS:
                if len(df) >= period: results[f'amt_ma{period}'] = TA.SMA(amount_df, period=period) # 使用 SMA
                else: results[f'amt_ma{period}'] = np.nan
            return pd.DataFrame(results, index=df.index)
        except Exception as e: logger.error(f"计算 Amount MA 时出错: {e}"); return None

    async def get_amount_ma_indicators(self, stock_code: str, kline_period: str, start_date=None, end_date=None):
        self.indicator_model = StockAmountMaIndicator
        self.indicator_name = "AmountMA"
        return await self.get_indicators(stock_code, kline_period, start_date, end_date)

    # --- Amount ROC ---
    async def calculate_and_save_amount_roc_stock(self, stock_code: str, kline_period: str):
        self.indicator_model = StockAmountRocIndicator
        self.indicator_name = "AmountROC"
        self.finta_required_columns = ['amount'] # 需要成交额
        await self.calculate_and_save_stock(stock_code, kline_period)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """计算 Amount ROC 指标"""
        if df is None or df.empty or 'amount' not in df.columns: return None
        try:
            results = {}
            amount_df = df[['amount']].copy()
            amount_df.rename(columns={'amount': 'close'}, inplace=True) # 欺骗 ROC 函数
            for period in FIB_PERIODS:
                if len(df) >= period + 1: results[f'aroc{period}'] = TA.ROC(amount_df, period=period)
                else: results[f'aroc{period}'] = np.nan
            return pd.DataFrame(results, index=df.index)
        except Exception as e: logger.error(f"计算 Amount ROC 时出错: {e}"); return None

    async def get_amount_roc_indicators(self, stock_code: str, kline_period: str, start_date=None, end_date=None):
        self.indicator_model = StockAmountRocIndicator
        self.indicator_name = "AmountROC"
        return await self.get_indicators(stock_code, kline_period, start_date, end_date)

    # --- VWAP ---
    async def calculate_and_save_vwap_stock(self, stock_code: str, kline_period: str):
        self.indicator_model = StockVwapIndicator
        self.indicator_name = "VWAP"
        # VWAP 需要 HLC 和 Volume
        self.finta_required_columns = ['high', 'low', 'close', 'volume']
        await self.calculate_and_save_stock(stock_code, kline_period)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """计算 VWAP 指标"""
        required = ['high', 'low', 'close', 'volume']
        if df is None or df.empty or not all(c in df.columns for c in required): return None
        try:
            # 确认 finta.TA.VWAP 的计算逻辑
            # 假设它计算的是每根 K 线的 VWAP (基于典型价格)
            results = {'vwap': TA.VWAP(df)}
            return pd.DataFrame(results, index=df.index)
        except Exception as e: logger.error(f"计算 VWAP 时出错: {e}"); return None

    async def get_vwap_indicators(self, stock_code: str, kline_period: str, start_date=None, end_date=None):
        self.indicator_model = StockVwapIndicator
        self.indicator_name = "VWAP"
        return await self.get_indicators(stock_code, kline_period, start_date, end_date)


# --- 服务实例化 (示例) ---
# high_priority_service = HighPriorityIndicatorService()
# medium_priority_service = MediumPriorityIndicatorService()
# low_priority_service = LowPriorityIndicatorService()

# --- 如何使用 (示例) ---
# async def run_calculation():
#     # 计算单个指标
#     await high_priority_service.calculate_and_save_ema_stock('000001', 'Day_qfq')
#     await medium_priority_service.calculate_and_save_dmi_stock('000001', 'Day_qfq')
#     await low_priority_service.calculate_and_save_sar_stock('000001', 'Day_qfq')
#     await low_priority_service.calculate_and_save_vwap_stock('000001', 'Day_qfq')

#     # 获取指标
#     ema_data = await high_priority_service.get_ema_indicators('000001', 'Day_qfq', end_date='2024-12-31')
#     print(ema_data[-5:]) # 打印最后5条

#     # 批量计算 (非常耗时，确保异步执行环境)
#     # await high_priority_service.calculate_and_save_all('Day_qfq')

# if __name__ == "__main__":
#     # 在 Django 环境下运行或设置 Django 环境
#     # import django
#     # django.setup()
#     # asyncio.run(run_calculation())
#     pass
