# services/indicator_calculate_services.py
import asyncio
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy.signal import find_peaks, peak_prominences

logger = logging.getLogger("services")

class IndicatorCalculator:
    """
    【V1.0 指标计算中心】
    - 核心职责: 封装所有独立的、纯粹的技术指标计算函数。
    - 设计原则: 本类不处理业务流程、数据获取或配置解析，仅专注于接收一个DataFrame并返回计算结果。
               这使得指标计算逻辑可以被独立测试和复用。
    """

    def __init__(self):
        """
        初始化指标计算器。
        目前不需要任何特定状态，但保留构造函数以便未来扩展。
        """
        pass

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
        【V1.1 异步优化版】计算 ATRR (平均真实波幅率)。
        - 指标含义: ATRR = ATR / Close，衡量波幅相对于价格的百分比，提供标准化的波动率度量。
        - 优化: 将同步的计算逻辑通过 asyncio.to_thread 移至工作线程执行，避免阻塞事件循环。
        """
        # 将所有前置检查放在同步函数外部，快速失败。
        required_cols = [high_col, low_col, close_col]
        if df is None or df.empty or not all(c in df.columns for c in required_cols):
            return None
        if len(df) < period:
            return None
        
        try:
            # 定义一个同步函数来封装所有计算逻辑。
            def _sync_atrr():
                # 1. 使用 pandas-ta 高效计算 ATR
                atr_series = ta.atr(high=df[high_col], low=df[low_col], close=df[close_col], length=period, append=False)
                if atr_series is None or atr_series.empty:
                    return None
                
                # 2. 计算 ATRR，并处理收盘价为0的情况以避免除零错误
                close_prices = df[close_col].replace(0, np.nan)
                atrr_series = atr_series / close_prices
                
                # 3. 将结果Series转换为DataFrame并返回
                return atrr_series.to_frame(name=f'ATRr_{period}')

            # 在独立的线程中异步执行同步计算函数。
            return await asyncio.to_thread(_sync_atrr)
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
                # pandas_ta 没有 atrn，我们先计算 atr
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
        【V1.2 代码简化版】计算布林带 (BBANDS) 及其宽度 (BBW) 和百分比B (%B)。
        - 核心修正: 对 pandas-ta 返回的 BBB% (带宽百分比) 列进行标准化，将其除以 100，转换为标准比率。
        - 优化: 移除了原代码中冗余的lambda表达式，使异步调用更清晰。
        """
        if df is None or df.empty or close_col not in df.columns:
            logger.warning(f"计算布林带缺少必要列: {close_col}。")
            return None
        if len(df) < period:
            logger.warning(f"数据行数 ({len(df)}) 不足以计算周期为 {period} 的布林带。")
            return None
        try:
            # 定义清晰的同步计算函数。
            def _sync_bbands_calculation():
                # 使用 ta.bbands() 直接调用，返回一个新的DataFrame
                bbands_df = ta.bbands(close=df[close_col], length=period, std=std_dev, append=False)
                if bbands_df is None or bbands_df.empty:
                    return None
                
                # 标准化带宽百分比列
                bbw_source_col = f'BBB_{period}_{std_dev:.1f}'
                if bbw_source_col in bbands_df.columns:
                    bbands_df[bbw_source_col] = bbands_df[bbw_source_col] / 100.0
                
                # 构建列名映射，以实现标准化命名
                rename_map = {
                    f'BBL_{period}_{std_dev:.1f}': f'BBL_{period}_{std_dev:.1f}{suffix}',
                    f'BBM_{period}_{std_dev:.1f}': f'BBM_{period}_{std_dev:.1f}{suffix}',
                    f'BBU_{period}_{std_dev:.1f}': f'BBU_{period}_{std_dev:.1f}{suffix}',
                    bbw_source_col: f'BBW_{period}_{std_dev:.1f}{suffix}',
                    f'BBP_{period}_{std_dev:.1f}': f'BBP_{period}_{std_dev:.1f}{suffix}'
                }
                
                result_df = bbands_df.rename(columns=rename_map)
                
                # 筛选出我们需要的列，确保返回结果的纯净性
                final_columns = list(rename_map.values())
                result_df = result_df[[col for col in final_columns if col in result_df.columns]]
                
                return result_df if not result_df.empty else None

            # 在线程中执行定义好的同步函数。
            return await asyncio.to_thread(_sync_bbands_calculation)
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
                # --- 统一使用 ta.kc() 直接调用，其行为更可预测 ---
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
        【V2.1 异步优化版】计算 CMF (蔡金货币流量)。
        - 指标含义: 衡量资金流入和流出的压力。CMF > 0 通常表示买方压力，CMF < 0 表示卖方压力。
        - 优化: 将同步的 pandas-ta 计算移至工作线程执行，避免阻塞事件循环。
        """
        required_cols = [high_col, low_col, close_col, volume_col]
        if df is None or df.empty or not all(col in df.columns for col in required_cols):
            logger.warning(f"计算 CMF (周期 {period}) 失败：输入DataFrame为空或缺少必需列 {required_cols}。")
            return None
        if len(df) < period:
            logger.warning(f"计算 CMF (周期 {period}) 失败：数据长度 {len(df)} 小于周期 {period}。")
            return None
        try:
            # 定义同步计算函数。
            def _sync_cmf():
                # 直接调用 pandas_ta，它会返回一个带有正确列名（如 'CMF_20'）的 Series
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
                # 将返回的 Series 转换为 DataFrame，以便上层服务进行合并
                return cmf_series.to_frame()

            # 在独立的线程中异步执行。
            return await asyncio.to_thread(_sync_cmf)
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
            if stoch_df is None or stoch_df.empty:
                logger.warning(f"KDJ (p={period}, sig={signal_period}, smooth={smooth_k_period}) 计算结果为空，可能数据量不足。")
                return None
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

    async def calculate_ma(self, df: pd.DataFrame, period: int, close_col='close') -> Optional[pd.DataFrame]:
        """[新增]计算 MA (简单移动平均线)"""
        if df is None or df.empty or close_col not in df.columns: return None
        if len(df) < period: return None
        try:
            def _sync_ma():
                # MA (Moving Average) 通常指的就是 SMA (Simple Moving Average)
                return ta.sma(close=df[close_col], length=period, append=False)
            ma_series = await asyncio.to_thread(_sync_ma)
            if ma_series is None or not isinstance(ma_series, pd.Series) or ma_series.empty: return None
            return pd.DataFrame({f'MA_{period}': ma_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 MA (周期 {period}) 时发生未知错误: {e}", exc_info=True)
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

            # ▼▼▼ 增加对 None 返回值的健壮性检查 ▼▼▼
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
        if len(df) <= period:
            logger.warning(f"数据行数 ({len(df)}) 不足以计算周期为 {period} 的 AROC。")
            return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_aroc():
                target_series = df[amount_col]
                # 直接调用 ta.roc 函数，传入 Series
                return ta.roc(close=target_series, length=period, append=False)
                
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

            # ▼▼▼ 增加对 None 返回值的健壮性检查 ▼▼▼
            if rsi_series is None or not isinstance(rsi_series, pd.Series) or rsi_series.empty:
                logger.warning(f"RSI (周期 {period}) 计算结果为空或无效，可能数据量不足。")
                return None
            
            #  在创建DataFrame时显式传入索引，更加安全
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
                # ▼▼▼: 核心修复，兼容Series和DataFrame返回值 ▼▼▼
                # 检查返回的是否是Series，如果是，则转换为DataFrame，以统一处理
                if isinstance(copp_df, pd.Series):
                    copp_df = copp_df.to_frame()

                # 现在 copp_df 肯定是DataFrame，可以安全地访问 .columns
                if not copp_df.columns[0].startswith('COPP'):
                    expected_name = f"COPP_{long_roc_period}_{short_roc_period}_{wma_period}"
                    actual_name = copp_df.columns[0]
                    copp_df.rename(columns={actual_name: expected_name}, inplace=True)
                    # print(f"    - [指标重命名] 已将列 '{actual_name}' 重命名为 '{expected_name}'")
                # ▲▲▲: 修改结束 ▲▲▲
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

    async def calculate_fibonacci_levels(self, df: pd.DataFrame, params: dict) -> Optional[pd.DataFrame]:
        """
        【V3.0 双引擎健壮版】计算斐波那契回撤水平。
        - 核心升级: 引入“优雅降级”机制。优先使用精密的find_peaks动态引擎，
                    如果动态引擎无法找到有效波段，则自动切换到基于滚动窗口的
                    备用引擎，确保永远能产出有效的斐波那契水平。
        """
        fib_params = params.get('params', {})
        if not params.get('enabled', False):
            return None

        # print("    - [斐波那契分析 V3.0] 启动双引擎分析...")
        
        try:
            from scipy.signal import find_peaks, peak_prominences
        except ImportError:
            logger.error("缺少 'scipy' 库，无法计算动态斐波那契水平。请运行 'pip install scipy'。")
            return None

        # --- 主引擎：动态波段识别 ---
        distance = fib_params.get('peak_distance', 13)
        prominence_ratio = fib_params.get('peak_prominence_ratio', 0.05)
        
        if 'close' not in df.columns:
            logger.error("斐波那契计算失败：DataFrame中缺少 'close' 列。")
            return None

        def _find_peaks_sync(data, prominence_series):
            candidate_indices, _ = find_peaks(data, distance=distance)
            if len(candidate_indices) == 0:
                return []
            actual_prominences, _, _ = peak_prominences(data, candidate_indices)
            custom_thresholds = prominence_series.iloc[candidate_indices]
            valid_mask = actual_prominences >= custom_thresholds.values
            return candidate_indices[valid_mask]

        peak_prominence_series = df['close'] * prominence_ratio
        peak_indices = await asyncio.to_thread(_find_peaks_sync, df['close'].values, peak_prominence_series)
        trough_indices = await asyncio.to_thread(_find_peaks_sync, -df['close'].values, peak_prominence_series)

        # --- 检查主引擎是否成功 ---
        if len(peak_indices) > 0 and len(trough_indices) > 0:
            # print("      -> [主引擎] 动态波段识别成功，正在计算...")
            
            temp_df = pd.DataFrame(index=df.index)
            temp_df['swing_high_price'] = np.nan
            temp_df.iloc[peak_indices, temp_df.columns.get_loc('swing_high_price')] = df['close'].iloc[peak_indices]
            temp_df['swing_high_price'].ffill(inplace=True)

            temp_df['swing_low_price'] = np.nan
            temp_df.iloc[trough_indices, temp_df.columns.get_loc('swing_low_price')] = df['close'].iloc[trough_indices]
            temp_df['swing_low_price'].ffill(inplace=True)
            
            temp_df['swing_high_date'] = pd.NaT
            temp_df.iloc[peak_indices, temp_df.columns.get_loc('swing_high_date')] = df.index[peak_indices]
            temp_df['swing_high_date'].ffill(inplace=True)
            
            temp_df['swing_low_date'] = pd.NaT
            temp_df.iloc[trough_indices, temp_df.columns.get_loc('swing_low_date')] = df.index[trough_indices]
            temp_df['swing_low_date'].ffill(inplace=True)

            is_uptrend_pullback = temp_df['swing_high_date'] > temp_df['swing_low_date']
            swing_range = abs(temp_df['swing_high_price'] - temp_df['swing_low_price'])

            result_df = pd.DataFrame(index=df.index)
            
            levels = fib_params.get('levels', [0.382, 0.5, 0.618])
            for level in levels:
                col_name = f'FIB_{level:.3f}'.replace('0.', '0_')
                retr_price = temp_df['swing_high_price'] - swing_range * level
                result_df[col_name] = np.where(is_uptrend_pullback, retr_price, np.nan)

            # print("    - [斐波那契分析 V3.0] 主引擎计算完成。")
            return result_df
        
        # --- 如果主引擎失败，则启动备用引擎 ---
        else:
            print("      -> [备用引擎] 动态波段识别失败，切换至滚动窗口模式。")
            result_df = pd.DataFrame(index=df.index)
            lookback = fib_params.get('lookback_period', 120)
            levels = fib_params.get('levels', [0.382, 0.5, 0.618])

            if not all(c in df.columns for c in ['high', 'low']):
                logger.error("斐波那契备用引擎计算失败：DataFrame中缺少 'high' 或 'low' 列。")
                return None

            rolling_high = df['high'].rolling(window=lookback).max()
            rolling_low = df['low'].rolling(window=lookback).min()
            price_range = rolling_high - rolling_low

            for level in levels:
                col_name = f'FIB_{level:.3f}'.replace('0.', '0_')
                result_df[col_name] = rolling_high - (price_range * level)
            
            # print("    - [斐波那契分析 V3.0] 备用引擎计算完成。")
            return result_df

    async def calculate_price_volume_ma_comparison(self, df: pd.DataFrame, params: dict) -> Optional[pd.DataFrame]:
        """
        【V1.7 · 职责净化版】计算价格/成交量与各自均线的比率。
        - 核心重构: 移除了方法内部对 'apply_on' 和后缀处理的冗余逻辑。
                    现在方法假定调用者已完成时间周期过滤，并传入了正确的、无后缀的DataFrame。
                    这使得本方法职责更单一，只专注于核心计算。
        """
        # 移除 enabled 检查，因为调用者已处理
        try:
            # 定义同步计算函数。
            def _sync_pv_ma_comparison():
                # 直接从 params 获取参数，不再需要复杂的后缀处理
                periods = params.get('periods', [])
                price_source_col = params.get('price_source')
                volume_source_col = params.get('volume_source')
                
                if not all([periods, price_source_col, volume_source_col]):
                    logger.warning("计算价比/量比缺少关键参数 (periods, price_source, volume_source)。")
                    return None
                
                result_df = pd.DataFrame(index=df.index)
                for p in periods:
                    # --- 计算价格与均线比 ---
                    price_ma_col = price_source_col if p == 1 else f'EMA_{p}'
                    if price_source_col in df.columns and price_ma_col in df.columns:
                        price_ma_series = df[price_ma_col].replace(0, np.nan)
                        ratio = df[price_source_col] / price_ma_series
                        result_df[f'price_vs_ma_{p}'] = ratio.fillna(1.0)
                    else:
                        logger.warning(f"计算 price_vs_ma_{p} 失败: 缺少列 {price_source_col} 或 {price_ma_col}")
                    # --- 计算成交量与均量比 ---
                    vol_ma_col = volume_source_col if p == 1 else f'VOL_MA_{p}'
                    if volume_source_col in df.columns and vol_ma_col in df.columns:
                        vol_ma_series = df[vol_ma_col].replace(0, np.nan)
                        ratio = df[volume_source_col] / vol_ma_series
                        result_df[f'volume_vs_ma_{p}'] = ratio.fillna(1.0)
                    else:
                        logger.warning(f"计算 volume_vs_ma_{p} 失败: 缺少列 {volume_source_col} 或 {vol_ma_col}")
                return result_df if not result_df.empty else None

            # 在独立的线程中异步执行。
            return await asyncio.to_thread(_sync_pv_ma_comparison)
        except Exception as e:
            logger.error(f"计算价格/成交量与均线比率时发生未知错误: {e}", exc_info=True)
            return None

    async def calculate_donchian(self, df: pd.DataFrame, period: int = 21, high_col='high', low_col='low') -> Optional[pd.DataFrame]:
        """计算唐奇安通道 (Donchian Channels)"""
        required_cols = [high_col, low_col]
        if df is None or df.empty or not all(c in df.columns for c in required_cols):
            return None
        if len(df) < period:
            return None
        try:
            def _sync_donchian():
                return ta.donchian(high=df[high_col], low=df[low_col], lower_length=period, upper_length=period, append=False)
            donchian_df = await asyncio.to_thread(_sync_donchian)
            if donchian_df is None or donchian_df.empty:
                return None
            return donchian_df
        except Exception as e:
            logger.error(f"计算 Donchian Channels (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_squeeze(self, df: pd.DataFrame, bb_period: int = 21, kc_period: int = 21, atr_period: int = 13, bb_std: float = 2.0, kc_mult: float = 1.5) -> Optional[pd.DataFrame]:
        """计算布林带与肯特纳通道的压缩 (Squeeze) 状态"""
        required_cols = ['high', 'low', 'close']
        if df is None or df.empty or not all(c in df.columns for c in required_cols):
            return None
        if len(df) < max(bb_period, kc_period, atr_period):
            return None
        try:
            def _sync_squeeze():
                # pandas-ta的squeeze指标直接计算了压缩状态
                return ta.squeeze(
                    high=df['high'], low=df['low'], close=df['close'],
                    bb_length=bb_period, bb_std=bb_std,
                    kc_length=kc_period, kc_scalar=kc_mult,
                    atr_length=atr_period,
                    append=False
                )
            squeeze_df = await asyncio.to_thread(_sync_squeeze)
            if squeeze_df is None or squeeze_df.empty:
                return None
            # 我们只需要SQZ_ON这一列，它是一个布尔值（0或1）
            squeeze_on_col = f'SQZ_ON'
            if squeeze_on_col in squeeze_df.columns:
                return squeeze_df[[squeeze_on_col]]
            return None
        except Exception as e:
            logger.error(f"计算 Squeeze (bb={bb_period}, kc={kc_period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_eom(self, df: pd.DataFrame, period: int = 13, high_col='high', low_col='low', volume_col='volume') -> Optional[pd.DataFrame]:
        """计算简易波动指标 (Ease of Movement)"""
        required_cols = [high_col, low_col, volume_col]
        if df is None or df.empty or not all(c in df.columns for c in required_cols):
            return None
        if len(df) < period:
            return None
        try:
            def _sync_eom():
                return ta.eom(high=df[high_col], low=df[low_col], volume=df[volume_col], length=period, append=False)
            eom_df = await asyncio.to_thread(_sync_eom)
            if eom_df is None or eom_df.empty:
                return None
            return eom_df
        except Exception as e:
            logger.error(f"计算 EOM (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_intraday_vwap_divergence_index(self, df_minute: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        【V1.3 · 命名规范修复版】计算日内VWAP偏离度积分指数。
        - 核心修复: 为输出列名增加 '_D' 后缀，以符合系统命名规范，确保下游模块能正确消费。
        """
        if df_minute is None or df_minute.empty:
            logger.warning("计算日内VWAP偏离指数失败：输入的分钟数据DataFrame为空。")
            return None
        close_col = next((c for c in df_minute.columns if c.startswith('close')), None)
        amount_col = next((c for c in df_minute.columns if c.startswith('amount')), None)
        volume_col = next((c for c in df_minute.columns if c.startswith('volume')), None)
        required_cols_map = {'close': close_col, 'amount': amount_col, 'volume': volume_col}
        missing_cols = [k for k, v in required_cols_map.items() if v is None]
        if missing_cols:
            logger.warning(f"计算日内VWAP偏离指数失败：分钟数据缺少基础列: {missing_cols}。")
            return None
        try:
            def _sync_calc():
                df = df_minute.copy()
                vwap_col = next((c for c in df.columns if c.startswith('vwap')), None)
                if vwap_col is None:
                    temp_amount = pd.to_numeric(df[amount_col], errors='coerce')
                    temp_volume = pd.to_numeric(df[volume_col], errors='coerce')
                    df['vwap_temp'] = temp_amount / temp_volume.replace(0, np.nan)
                    df['vwap_temp'].fillna(method='ffill', inplace=True)
                    vwap_col = 'vwap_temp'
                vwap_deviation = (df[close_col] - df[vwap_col]) / df[vwap_col].replace(0, np.nan)
                daily_integral = vwap_deviation.resample('D').sum()
                # [代码修改开始]
                result_df = pd.DataFrame({'intraday_vwap_div_index_D': daily_integral})
                # [代码修改结束]
                return result_df.dropna()
            return await asyncio.to_thread(_sync_calc)
        except Exception as e:
            logger.error(f"计算日内VWAP偏离指数时发生错误: {e}", exc_info=True)
            return None

    async def calculate_counterparty_exhaustion_index(self, df_minute: pd.DataFrame, efficiency_window: int = 21) -> Optional[pd.DataFrame]:
        """
        【V2.2 · 解耦聚合修复版】计算对手盘衰竭指数。
        - 核心修复: 彻底解决Pandas聚合逻辑陷阱。将复杂的agg指令分解为多个原子操作：
                    1. 单独聚合求和项。
                    2. 单独聚合计算日内涨跌幅。
                    3. 合并结果。
                    这确保了聚合逻辑的清晰性和健壮性，根除了KeyError。
        """
        if df_minute is None or df_minute.empty or len(df_minute) < 10:
            return None
        open_col = next((c for c in df_minute.columns if c.startswith('open')), None)
        high_col = next((c for c in df_minute.columns if c.startswith('high')), None)
        low_col = next((c for c in df_minute.columns if c.startswith('low')), None)
        close_col = next((c for c in df_minute.columns if c.startswith('close')), None)
        volume_col = next((c for c in df_minute.columns if c.startswith('volume')), None)
        
        required_cols = [open_col, high_col, low_col, close_col, volume_col]
        if not all(required_cols):
            missing = [name for name, col in zip(['open', 'high', 'low', 'close', 'volume'], required_cols) if col is None]
            logger.warning(f"计算对手盘衰竭指数失败：分钟数据缺少基础列: {missing}。")
            return None
            
        try:
            def _sync_calc():
                df = df_minute.copy()
                df['directional_thrust'] = (df[close_col] - df[open_col]) * df[volume_col]
                df['total_energy'] = (df[high_col] - df[low_col]) * df[volume_col]
                
                # [代码修改开始]
                # 步骤1: 先聚合简单的求和项
                daily_sums = df.resample('D').agg({
                    'directional_thrust': 'sum',
                    'total_energy': 'sum'
                })
                daily_sums.rename(columns={'directional_thrust': 'daily_thrust', 'total_energy': 'daily_energy'}, inplace=True)

                # 步骤2: 单独计算日内真实涨跌幅
                daily_ohlc = df[close_col].resample('D').ohlc()
                # 确保 ohlc 结果不为空
                if daily_ohlc.empty:
                    return None
                daily_ohlc['pct_change'] = (daily_ohlc['close'] / daily_ohlc['open'].replace(0, np.nan) - 1).fillna(0)
                
                # 步骤3: 合并结果，形成最终的日级别聚合DataFrame
                daily_agg = daily_sums.join(daily_ohlc[['pct_change']], how='inner')
                # [代码修改结束]
                
                daily_agg['conversion_efficiency'] = (daily_agg['daily_thrust'] / daily_agg['daily_energy'].replace(0, np.nan)).fillna(0)
                
                efficiency_zscore = (daily_agg['conversion_efficiency'] - daily_agg['conversion_efficiency'].rolling(efficiency_window).mean()) / (daily_agg['conversion_efficiency'].rolling(efficiency_window).std() + 1e-9)
                
                is_buying_exhaustion = (daily_agg['pct_change'] > 0) & (efficiency_zscore < -0.5)
                is_selling_exhaustion = (daily_agg['pct_change'] < 0) & (efficiency_zscore > 0.5)
                
                exhaustion_index = (is_selling_exhaustion.astype(int) - is_buying_exhaustion.astype(int)).astype(float)
                
                return pd.DataFrame({'counterparty_exhaustion_index_D': exhaustion_index})

            return await asyncio.to_thread(_sync_calc)
        except Exception as e:
            logger.error(f"计算对手盘衰竭指数(V2.2)时发生错误: {e}", exc_info=True)
            return None

    async def calculate_breakout_quality_score(self, df_daily: pd.DataFrame, params: dict) -> Optional[pd.DataFrame]:
        """
        【V2.1 · 指挥链修复版】计算突破质量分。
        - 核心修复: 修正了依赖列的查找逻辑。由于此方法在上游被调用时，传入的 df_daily 的列名后缀已被剥离，
                    因此这里必须查找无后缀的列名，以确保能正确获取数据。
        """
        if df_daily is None or df_daily.empty:
            return None
        try:
            def _sync_calc():
                df = df_daily.copy()
                weights = params.get('weights', {'volume': 0.2, 'driver': 0.3, 'price_action': 0.1, 'chips': 0.2, 'efficiency': 0.2})
                
                # [代码修改开始]
                # 核心修复：此方法被调用时，传入的df_daily的列名后缀已被上游剥离。
                # 因此，这里必须查找无后缀的列名。
                required_cols = [
                    'volume', 'VOL_MA_21', 'main_force_flow_directionality',
                    'open', 'high', 'low', 'close',
                    'total_winner_rate', 'dominant_peak_solidity', 'VPA_EFFICIENCY'
                ]
                # [代码修改结束]

                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"调试信息: 计算突破质量分(V2.1)失败，缺少必要列: {missing_cols}。可用列: {df.columns.tolist()}")
                    return None
                
                # [代码修改开始]
                # 维度一：能量输入 (0-1分)
                volume_ratio = df['volume'] / df['VOL_MA_21'].replace(0, np.nan)
                score_volume = volume_ratio.rolling(60).rank(pct=True).fillna(0.5)

                # 维度二：主导力量 (天然是-1到1分，映射到0-1)
                score_driver = (df['main_force_flow_directionality'].fillna(0) + 1) / 2

                # 维度三：价格形态 (0-1分)
                price_range = (df['high'] - df['low']).replace(0, np.nan)
                score_price_action = ((df['close'] - df['open']) / price_range).fillna(0).clip(0, 1)

                # 维度四：筹码结构 (0-1分)
                winner_rate_gain = df['total_winner_rate'].diff().fillna(0)
                chip_breakthrough_eff = winner_rate_gain * df['dominant_peak_solidity']
                score_chips = chip_breakthrough_eff.rolling(60).rank(pct=True).fillna(0.5)

                # 维度五：攻击效率 (0-1分)
                score_efficiency = df['VPA_EFFICIENCY'].rolling(60).rank(pct=True).fillna(0.5)
                # [代码修改结束]
                
                quality_score = (
                    score_volume * weights['volume'] +
                    score_driver * weights['driver'] +
                    score_price_action * weights['price_action'] +
                    score_chips * weights['chips'] +
                    score_efficiency * weights['efficiency']
                )
                return pd.DataFrame({'breakout_quality_score_D': quality_score})
            return await asyncio.to_thread(_sync_calc)
        except Exception as e:
            logger.error(f"计算突破质量分(V2.1)时发生错误: {e}", exc_info=True)
            return None







