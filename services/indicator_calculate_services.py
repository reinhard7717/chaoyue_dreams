# services/indicator_calculate_services.py
import asyncio
import logging
from typing import Dict, List, Optional
import nolds
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
    async def calculate_atr(self, df: pd.DataFrame, period: int = 14, high_col='high', low_col='low', close_col='close', suffix: str = '') -> Optional[pd.DataFrame]:
        """【V1.1 · 命名净化版】计算 ATR (平均真实波幅)"""
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
            # 返回纯净的、不带后缀的列名
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
        【V1.3 · 接口契约修复版】计算布林带 (BBANDS) 及其宽度 (BBW) 和百分比B (%B)。
        - 核心修复: 确保使用调用方传入的 close_col 进行计算。
        """
        if df is None or df.empty or close_col not in df.columns:
            logger.warning(f"计算布林带缺少必要列: {close_col}。")
            return None
        if len(df) < period:
            logger.warning(f"数据行数 ({len(df)}) 不足以计算周期为 {period} 的布林带。")
            return None
        try:
            def _sync_bbands_calculation():
                # 使用调用方传入的 close_col
                bbands_df = ta.bbands(close=df[close_col], length=period, std=std_dev, append=False)
                if bbands_df is None or bbands_df.empty:
                    return None
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
                final_columns = list(rename_map.values())
                result_df = result_df[[col for col in final_columns if col in result_df.columns]]
                return result_df if not result_df.empty else None
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
    async def calculate_cmf(self, df: pd.DataFrame, period: int = 20, high_col='high', low_col='low', close_col='close', volume_col='volume', suffix: str = '') -> Optional[pd.DataFrame]:
        """
        【V2.2 · 接口健壮性修复版】计算 CMF (蔡金货币流量)。
        - 核心修复: 在函数签名中添加 suffix='' 参数，以优雅地接收并忽略上游可能泄漏的参数，防止TypeError。
        """
        required_cols = [high_col, low_col, close_col, volume_col]
        if df is None or df.empty or not all(col in df.columns for col in required_cols):
            logger.warning(f"计算 CMF (周期 {period}) 失败：输入DataFrame为空或缺少必需列 {required_cols}。")
            return None
        if len(df) < period:
            logger.warning(f"计算 CMF (周期 {period}) 失败：数据长度 {len(df)} 小于周期 {period}。")
            return None
        try:
            def _sync_cmf():
                cmf_series = ta.cmf(
                    high=df[high_col],
                    low=df[low_col],
                    close=df[close_col],
                    volume=df[volume_col],
                    length=period,
                    append=False
                )
                if cmf_series is None or cmf_series.empty:
                    logger.warning(f"计算 CMF (周期 {period}) 返回了空结果。")
                    return None
                return cmf_series.to_frame()
            return await asyncio.to_thread(_sync_cmf)
        except Exception as e:
            logger.error(f"计算 CMF (周期 {period}) 时发生未知异常: {e}", exc_info=True)
            return None
    async def calculate_dma(self, df: pd.DataFrame, smooth_factor_series: pd.Series, close_col: str = 'close') -> Optional[pd.DataFrame]:
        """
        【V1.0】计算 DMA (动态移动平均线)。
        - 核心逻辑: DMA 的平滑因子是一个动态变化的 Series，而不是固定的周期。
        - 数学公式: DMA = (CLOSE * smooth_factor + REF(DMA,1) * (1 - smooth_factor))
        """
        if df is None or df.empty or close_col not in df.columns:
            logger.warning(f"计算 DMA 失败：输入 DataFrame 为空或缺少 '{close_col}' 列。")
            return None
        if smooth_factor_series is None or smooth_factor_series.empty:
            logger.warning(f"计算 DMA 失败：平滑因子 Series 为空。")
            return None
        # 确保 smooth_factor_series 的索引与 df 对齐
        smooth_factor_series = smooth_factor_series.reindex(df.index, fill_value=0).clip(0, 1) # 平滑因子应在 [0, 1] 之间
        try:
            def _sync_dma():
                dma_series = pd.Series(np.nan, index=df.index)
                # 初始值
                dma_series.iloc[0] = df[close_col].iloc[0]
                for i in range(1, len(df)):
                    sf = smooth_factor_series.iloc[i]
                    if pd.isna(sf) or sf == 0: # 如果平滑因子为0，则DMA保持不变
                        dma_series.iloc[i] = dma_series.iloc[i-1]
                    else:
                        dma_series.iloc[i] = df[close_col].iloc[i] * sf + dma_series.iloc[i-1] * (1 - sf)
                return dma_series
            dma_series = await asyncio.to_thread(_sync_dma)
            if dma_series is None or dma_series.empty:
                logger.warning(f"DMA 计算结果为空。")
                return None
            return pd.DataFrame({'DMA': dma_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 DMA (close_col={close_col}) 出错: {e}", exc_info=True)
            return None
    async def calculate_atan_ma_angle(self, df: pd.DataFrame, ma_col_base: str, timeframe_key: str) -> Optional[pd.DataFrame]:
        """
        【V1.1】计算均线的角度 (ATAN)。
        - 核心逻辑: 将均线的日间变化率转换为角度，反映均线的陡峭程度。
        - 数学公式: ATAN((MA / REF(MA,1) - 1) * 100) * 180 / PI
        - 【修复】返回未带时间框架后缀的列名，由上层统一添加。
        """
        ma_col_full = f"{ma_col_base}_{timeframe_key}"
        if df is None or df.empty or ma_col_full not in df.columns:
            logger.warning(f"计算 ATAN 均线角度失败：输入 DataFrame 为空或缺少 '{ma_col_full}' 列。")
            return None
        try:
            def _sync_atan_angle():
                ma_series = df[ma_col_full]
                prev_ma = ma_series.shift(1).replace(0, np.nan)
                change_rate = (ma_series / prev_ma - 1) * 100
                angle_series = np.arctan(change_rate) * 180 / np.pi
                return angle_series
            angle_series = await asyncio.to_thread(_sync_atan_angle)
            if angle_series is None or angle_series.empty:
                logger.warning(f"ATAN 均线角度计算结果为空。")
                return None
            return pd.DataFrame({f'ATAN_ANGLE_{ma_col_base}': angle_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 ATAN 均线角度 (ma_col_base={ma_col_base}) 出错: {e}", exc_info=True)
            return None
    async def calculate_dmi(self, df: pd.DataFrame, period: int = 14, high_col='high', low_col='low', close_col='close', suffix: str = '') -> Optional[pd.DataFrame]:
        """【V1.2 · f-string修复版】计算 DMI (动向指标), 包括 PDI (+DI), NDI (-DI), ADX"""
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
            # 修复 f-string 遗忘错误
            final_cols = [f'PDI_{period}', f'NDI_{period}', f'ADX_{period}']
            # 确保所有列都存在再进行选择
            existing_cols = [col for col in final_cols if col in result_df.columns]
            if not existing_cols:
                return None
            return result_df[existing_cols]
        except Exception as e:
            logger.error(f"计算 DMI (周期 {period}) 出错: {e}", exc_info=True)
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
    async def calculate_ma(self, df: pd.DataFrame, period: int, close_col='close', suffix: str = '') -> Optional[pd.DataFrame]:
        """【V1.1 · 命名净化版】计算 MA (简单移动平均线)"""
        if df is None or df.empty or close_col not in df.columns: return None
        if len(df) < period: return None
        try:
            def _sync_ma():
                return ta.sma(close=df[close_col], length=period, append=False)
            ma_series = await asyncio.to_thread(_sync_ma)
            if ma_series is None or not isinstance(ma_series, pd.Series) or ma_series.empty: return None
            # 返回纯净的、不带后缀的列名
            return pd.DataFrame({f'MA_{period}': ma_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 MA (周期 {period}) 时发生未知错误: {e}", exc_info=True)
            return None
    async def calculate_ma_velocity_acceleration(self, df: pd.DataFrame, ma_col_base: str, timeframe_key: str, ema_period: int = 3, sma_period: int = 3) -> Optional[pd.DataFrame]:
        """
        【V1.1】计算均线的速度和加速度。
        - 核心逻辑: 速度是均线变化率的平滑，加速度是速度的平滑变化。
        - 数学公式:
          速度 = SMA(EMA((MA - REF(MA,1))/REF(MA,1), ema_period) * 100, sma_period, 1)
          加速度 = EMA((速度 - REF(速度,1)), ema_period)
        - 【修复】返回未带时间框架后缀的列名，由上层统一添加。
        """
        ma_col_full = f"{ma_col_base}_{timeframe_key}"
        if df is None or df.empty or ma_col_full not in df.columns:
            logger.warning(f"计算均线速度加速度失败：输入 DataFrame 为空或缺少 '{ma_col_full}' 列。")
            return None
        try:
            def _sync_calc():
                ma_series = df[ma_col_full]
                prev_ma = ma_series.shift(1).replace(0, np.nan)
                ma_change_rate = (ma_series / prev_ma - 1) * 100
                velocity_ema = ta.ema(close=ma_change_rate, length=ema_period, append=False)
                velocity_series = ta.sma(close=velocity_ema, length=sma_period, append=False)
                prev_velocity = velocity_series.shift(1)
                acceleration_series = ta.ema(close=(velocity_series - prev_velocity), length=ema_period, append=False)
                results_df = pd.DataFrame({
                    f'MA_VELOCITY_{ma_col_base}': velocity_series,
                    f'MA_ACCELERATION_{ma_col_base}': acceleration_series
                }, index=df.index)
                return results_df
            results_df = await asyncio.to_thread(_sync_calc)
            if results_df is None or results_df.empty:
                logger.warning(f"均线速度加速度计算结果为空。")
                return None
            return results_df
        except Exception as e:
            logger.error(f"计算均线速度加速度 (ma_col_base={ma_col_base}) 出错: {e}", exc_info=True)
            return None
    async def calculate_ema(self, df: pd.DataFrame, period: int, close_col='close', suffix: str = '') -> Optional[pd.DataFrame]:
        """【V1.1 · 命名净化版】计算 EMA (指数移动平均线)"""
        if df is None or df.empty or close_col not in df.columns: return None
        if len(df) < period: return None
        try:
            def _sync_ema():
                return ta.ema(close=df[close_col], length=period, append=False)
            ema_series = await asyncio.to_thread(_sync_ema)
            if ema_series is None or not isinstance(ema_series, pd.Series) or ema_series.empty: return None
            # 返回纯净的、不带后缀的列名
            return pd.DataFrame({f'EMA_{period}': ema_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 EMA (周期 {period}) 时发生未知错误: {e}", exc_info=True)
            return None
    async def calculate_zigzag(self, df: pd.DataFrame, period: int = 3, percent: float = 5.0, close_col: str = 'close') -> Optional[pd.DataFrame]:
        """
        【V1.0】计算 ZIGZAG 指标。
        - 核心逻辑: 识别价格的主要趋势和转折点，过滤掉小于指定百分比的波动。
        - 注意: pandas_ta 没有直接的 zigzag 实现，这里需要手动实现或使用其他库。
                为了简化，这里将使用一个基于百分比回撤的简化逻辑。
        """
        if df is None or df.empty or close_col not in df.columns:
            logger.warning(f"计算 ZIGZAG 失败：输入 DataFrame 为空或缺少 '{close_col}' 列。")
            return None
        if len(df) < period:
            logger.warning(f"数据行数 ({len(df)}) 不足以计算周期为 {period} 的 ZIGZAG。")
            return None
        try:
            def _sync_zigzag():
                prices = df[close_col].values
                zigzag_points = [prices[0]]
                zigzag_indices = [0]
                last_peak = prices[0]
                last_peak_idx = 0
                last_trough = prices[0]
                last_trough_idx = 0
                trend = 0 # 0: flat, 1: up, -1: down
                for i in range(1, len(prices)):
                    current_price = prices[i]
                    if trend == 0:
                        if current_price >= last_peak * (1 + percent / 100):
                            trend = 1
                            last_peak = current_price
                            last_peak_idx = i
                        elif current_price <= last_trough * (1 - percent / 100):
                            trend = -1
                            last_trough = current_price
                            last_trough_idx = i
                    elif trend == 1: # Up trend
                        if current_price > last_peak:
                            last_peak = current_price
                            last_peak_idx = i
                        elif current_price <= last_peak * (1 - percent / 100):
                            # New trough detected
                            zigzag_points.append(last_peak)
                            zigzag_indices.append(last_peak_idx)
                            last_trough = current_price
                            last_trough_idx = i
                            trend = -1
                    elif trend == -1: # Down trend
                        if current_price < last_trough:
                            last_trough = current_price
                            last_trough_idx = i
                        elif current_price >= last_trough * (1 + percent / 100):
                            # New peak detected
                            zigzag_points.append(last_trough)
                            zigzag_indices.append(last_trough_idx)
                            last_peak = current_price
                            last_peak_idx = i
                            trend = 1
                # Add the last point
                if trend == 1:
                    zigzag_points.append(last_peak)
                    zigzag_indices.append(last_peak_idx)
                elif trend == -1:
                    zigzag_points.append(last_trough)
                    zigzag_indices.append(last_trough_idx)
                zigzag_series = pd.Series(np.nan, index=df.index)
                for i in range(len(zigzag_points)):
                    zigzag_series.iloc[zigzag_indices[i]] = zigzag_points[i]
                # 前向填充，使每个点都有一个zigzag值，代表其所属的zigzag段
                zigzag_series = zigzag_series.ffill().bfill()
                return zigzag_series
            zigzag_series = await asyncio.to_thread(_sync_zigzag)
            if zigzag_series is None or zigzag_series.empty:
                logger.warning(f"ZIGZAG (period={period}, percent={percent}) 计算结果为空。")
                return None
            return pd.DataFrame({f'ZIG_{period}_{percent}': zigzag_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 ZIGZAG (period={period}, percent={percent}) 出错: {e}", exc_info=True)
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
    async def calculate_macd(self, df: pd.DataFrame, period_fast: int = 12, period_slow: int = 26, signal_period: int = 9, close_col='close', suffix: str = '') -> Optional[pd.DataFrame]:
        """【V1.1 · 命名净化版】计算移动平均收敛散度 (MACD)"""
        if df is None or df.empty or close_col not in df.columns:
            return None
        if len(df) < period_slow + signal_period:
            return None
        try:
            def _sync_macd():
                return ta.macd(close=df[close_col], fast=period_fast, slow=period_slow, signal=signal_period, append=False)
            macd_df = await asyncio.to_thread(_sync_macd)
            if macd_df is None or macd_df.empty:
                logger.warning(f"MACD (f={period_fast},s={period_slow},sig={signal_period}) 计算结果为空，可能数据量不足。")
                return None
            # pandas-ta返回的列名已经包含了周期，是纯净的，直接返回即可
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
    async def calculate_obv(self, df: pd.DataFrame, close_col='close', volume_col='volume', suffix: str = '') -> Optional[pd.DataFrame]:
        """【V1.1 · 命名净化版】计算 OBV (能量潮指标)"""
        if df is None or df.empty or not all(c in df.columns for c in [close_col, volume_col]): return None
        try:
            def _sync_obv():
                return ta.obv(close=df[close_col], volume=df[volume_col], append=False)
            obv_series = await asyncio.to_thread(_sync_obv)
            if obv_series is None or obv_series.empty: return None
            # 返回纯净的、不带后缀的列名
            return pd.DataFrame({'OBV': obv_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 OBV 出错: {e}", exc_info=True)
            return None
    async def calculate_roc(self, df: pd.DataFrame, period: int = 12, close_col='close', suffix: str = '') -> Optional[pd.DataFrame]:
        """【V1.1 · 命名净化版】计算 ROC (价格变化率)"""
        if df is None or df.empty or close_col not in df.columns: return None
        if len(df) <= period: return None
        try:
            def _sync_roc():
                return ta.roc(close=df[close_col], length=period, append=False)
            roc_series = await asyncio.to_thread(_sync_roc)
            if roc_series is None or roc_series.empty: return None
            # 返回纯净的、不带后缀的列名
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
    async def calculate_rsi(self, df: pd.DataFrame, period: int, close_col='close', suffix: str = '') -> Optional[pd.DataFrame]:
        """【V1.1 · 命名净化版】计算相对强弱指数 (RSI)"""
        if df is None or df.empty or close_col not in df.columns:
            return None
        if len(df) < period:
            return None
        try:
            def _sync_rsi():
                return ta.rsi(close=df[close_col], length=period, append=False)
            rsi_series = await asyncio.to_thread(_sync_rsi)
            if rsi_series is None or not isinstance(rsi_series, pd.Series) or rsi_series.empty:
                logger.warning(f"RSI (周期 {period}) 计算结果为空或无效，可能数据量不足。")
                return None
            # 返回纯净的、不带后缀的列名
            return pd.DataFrame({f'RSI_{period}': rsi_series}, index=df.index)
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
    async def calculate_vol_ma(self, df: pd.DataFrame, period: int = 20, volume_col='volume', suffix: str = '') -> Optional[pd.DataFrame]:
        """【V1.1 · 命名净化版】计算成交量的移动平均线 (VOL_MA)"""
        if df is None or df.empty or volume_col not in df.columns: return None
        if len(df) < period: return None
        try:
            def _sync_vol_ma():
                return df[volume_col].rolling(window=period, min_periods=max(1, int(period*0.5))).mean()
            vol_ma_series = await asyncio.to_thread(_sync_vol_ma)
            # 返回纯净的、不带后缀的列名
            return pd.DataFrame({f'VOL_MA_{period}': vol_ma_series})
        except Exception as e:
            logger.error(f"计算 VOL_MA (周期 {period}) 出错: {e}", exc_info=True)
            return None
    async def calculate_vwap(self, df: pd.DataFrame, anchor: Optional[str] = None, high_col='high', low_col='low', close_col='close', volume_col='volume', suffix: str = '') -> Optional[pd.DataFrame]:
        """
        【V1.2 · 接口契约修复版】计算 VWAP (成交量加权平均价)。
        - 核心修复: 确保使用调用方传入的 OHLCV 列名进行计算。
        """
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col, volume_col]):
            logger.warning(f"计算 VWAP (anchor={anchor}) 时缺少必要的列。")
            return None
        processed_anchor = anchor
        if anchor and str(anchor).isdigit():
            processed_anchor = f"{anchor}T"
        try:
            def _sync_vwap():
                # 使用调用方传入的列名
                return df.ta.vwap(high=df[high_col], low=df[low_col], close=df[close_col], volume=df[volume_col], anchor=processed_anchor, append=False)
            vwap_series = await asyncio.to_thread(_sync_vwap)
            if vwap_series is None or vwap_series.empty: return None
            original_name = vwap_series.name
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
    async def calculate_coppock(self, df: pd.DataFrame, long_roc_period: int = 26, short_roc_period: int = 13, wma_period: int = 10, close_col: str = 'close') -> Optional[pd.DataFrame]:
        """
        【V1.4 · 接口契约修复版】计算 Coppock Curve (COPP) 指标。
        - 核心修复: 确保使用调用方传入的 close_col 进行计算。
        """
        if df is None or df.empty or close_col not in df.columns:
            return None
        try:
            # 使用调用方传入的 close_col
            copp_df = df.ta.coppock(
                close=df[close_col],
                length=wma_period,
                fast=short_roc_period,
                slow=long_roc_period,
                append=False
            )
            if copp_df is not None and not copp_df.empty:
                if isinstance(copp_df, pd.Series):
                    copp_df = copp_df.to_frame()
                if not copp_df.columns[0].startswith('COPP'):
                    suffix = close_col[close_col.rfind('_'):] if '_' in close_col else ''
                    expected_name = f"COPP_{long_roc_period}_{short_roc_period}_{wma_period}{suffix}"
                    actual_name = copp_df.columns[0]
                    copp_df.rename(columns={actual_name: expected_name}, inplace=True)
                return copp_df
        except Exception as e:
            if "data length" in str(e).lower() or "inputs are all nan" in str(e).lower():
                logger.warning(f"数据行数 ({len(df)}) 不足以计算 Coppock Curve(long={long_roc_period}, short={short_roc_period}, wma={wma_period})。")
            else:
                logger.error(f"计算 Coppock Curve 时发生未知错误: {type(e).__name__}: {e}", exc_info=False)
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
    async def calculate_bias(self, df: pd.DataFrame, period: int = 20, close_col='close', suffix: str = '') -> Optional[pd.DataFrame]:
        """
        【V1.4 · 命名净化版】计算 BIAS，并强制重命名列以符合系统标准。
        """
        if df is None or df.empty or close_col not in df.columns:
            logger.warning(f"BIAS计算失败：输入的DataFrame为空或缺少'{close_col}'列。")
            return None
        if len(df) < period:
            logger.warning(f"BIAS计算失败：数据长度 {len(df)} 小于所需周期 {period}。")
            return None
        try:
            def _sync_bias() -> Optional[pd.Series]:
                return df.ta.bias(close=df[close_col], length=period, append=False)
            bias_series = await asyncio.to_thread(_sync_bias)
            if bias_series is None or bias_series.empty:
                logger.warning(f"pandas_ta.bias 未能为周期 {period} 生成有效结果。")
                return None
            # 返回纯净的、不带后缀的列名
            target_col_name = f"BIAS_{period}"
            bias_series.name = target_col_name
            result_df = pd.DataFrame(bias_series)
            return result_df
        except Exception as e:
            logger.error(f"计算 BIAS (period={period}) 时发生未知错误: {e}", exc_info=True)
            return None
    async def calculate_fibonacci_levels(self, df: pd.DataFrame, params: dict, suffix: str = '') -> Optional[pd.DataFrame]:
        """
        【V1.2 · 接口净化版】计算斐波那契回撤/扩展位。
        - 移除 timeframe_key 参数，改为接收一个通用的 suffix，由调用方构建。
        """
        result_df = pd.DataFrame(index=df.index)
        periods = params.get('periods', [])
        # 使用传入的后缀构建列名
        high_col = f"high{suffix}"
        low_col = f"low{suffix}"
        if high_col not in df.columns or low_col not in df.columns:
            logger.warning(f"斐波那契水平计算失败：缺少源列 {high_col} 或 {low_col}")
            return None
        fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        for period in periods:
            rolling_high = df[high_col].rolling(window=period).max()
            rolling_low = df[low_col].rolling(window=period).min()
            price_range = rolling_high - rolling_low
            for level in fib_levels:
                level_name = str(level).replace('.', '_')
                # 在输出列名中也使用后缀
                result_df[f'fib_{level_name}_support_{period}{suffix}'] = rolling_low + (price_range * level)
                result_df[f'fib_{level_name}_resistance_{period}{suffix}'] = rolling_high - (price_range * level)
        return result_df
    async def calculate_price_volume_ma_comparison(self, df: pd.DataFrame, params: dict, suffix: str = '') -> Optional[pd.DataFrame]:
        """
        【V4.2 · 接口净化版】计算价格/成交量与各自均线的比率。
        - 移除 timeframe_key 参数，改为接收一个通用的 suffix，由调用方构建。
        """
        result_df = pd.DataFrame(index=df.index)
        periods = params.get('periods', [])
        # 使用传入的后缀构建列名
        price_source_col = f"{params.get('price_source', 'close')}{suffix}"
        volume_source_col = f"{params.get('volume_source', 'volume')}{suffix}"
        if price_source_col not in df.columns or volume_source_col not in df.columns:
            logger.warning(f"价格/成交量均线比率计算失败：缺少源列 {price_source_col} 或 {volume_source_col}")
            return None
        for period in periods:
            # 在查找依赖列和构建输出列时，都使用后缀
            price_ma_col = f"EMA_{period}{suffix}"
            vol_ma_col = f"VOL_MA_{period}{suffix}"
            if price_ma_col in df.columns:
                price_ratio = df[price_source_col] / df[price_ma_col].replace(0, np.nan)
                result_df[f'price_vs_ma_{period}_ratio{suffix}'] = price_ratio.fillna(1.0)
            if vol_ma_col in df.columns:
                volume_ratio = df[volume_source_col] / df[vol_ma_col].replace(0, np.nan)
                result_df[f'volume_vs_ma_{period}_ratio{suffix}'] = volume_ratio.fillna(1.0)
        return result_df
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
        【V1.4 · 命名规范修复版】计算日内VWAP偏离度积分指数。
        - 核心修复: 返回不带 '_D' 后缀的列名，以符合系统命名规范，确保下游模块能正确消费。
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
                # 返回不带 '_D' 后缀的列名
                result_df = pd.DataFrame({'intraday_vwap_div_index': daily_integral})
                return result_df.dropna()
            return await asyncio.to_thread(_sync_calc)
        except Exception as e:
            logger.error(f"计算日内VWAP偏离指数时发生错误: {e}", exc_info=True)
            return None
    async def calculate_counterparty_exhaustion_index(self, df_minute: pd.DataFrame, efficiency_window: int = 21) -> Optional[pd.DataFrame]:
        """
        【V2.3 · 解耦聚合与命名修复版】计算对手盘衰竭指数。
        - 核心修复: 返回不带 '_D' 后缀的列名，以符合系统命名规范，确保下游模块能正确消费。
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
                daily_agg['conversion_efficiency'] = (daily_agg['daily_thrust'] / daily_agg['daily_energy'].replace(0, np.nan)).fillna(0)
                efficiency_zscore = (daily_agg['conversion_efficiency'] - daily_agg['conversion_efficiency'].rolling(efficiency_window).mean()) / (daily_agg['conversion_efficiency'].rolling(efficiency_window).std() + 1e-9)
                is_buying_exhaustion = (daily_agg['pct_change'] > 0) & (efficiency_zscore < -0.5)
                is_selling_exhaustion = (daily_agg['pct_change'] < 0) & (efficiency_zscore > 0.5)
                # 返回不带 '_D' 后缀的列名
                exhaustion_index = (is_selling_exhaustion.astype(int) - is_buying_exhaustion.astype(int)).astype(float)
                return pd.DataFrame({'counterparty_exhaustion_index': exhaustion_index})
            return await asyncio.to_thread(_sync_calc)
        except Exception as e:
            logger.error(f"计算对手盘衰竭指数(V2.3)时发生错误: {e}", exc_info=True)
            return None
    async def calculate_breakout_quality_score(self, df_daily: pd.DataFrame, params: dict) -> Optional[pd.DataFrame]:
        """
        【V2.7 · 细粒度斐波那契突破质量增强版】计算突破质量分。
        - 核心修复: 遵循“纯净计算”原则，返回不带任何后缀的列名 'breakout_quality_score'，
                      将命名标准化的责任完全交由上游编排器处理。
        - 细粒度增强: 全面利用细粒度买卖方数据、订单簿流动性、欺骗指数等，提升评估精度。
        - 周期调整: 调整滚动窗口为斐波那契数列。
        """
        if df_daily is None or df_daily.empty:
            return None
        try:
            def _sync_calc():
                df = df_daily.copy()
                weights = params.get('weights', {'volume': 0.2, 'driver': 0.3, 'price_action': 0.1, 'chips': 0.2, 'efficiency': 0.2, 'purity_penalty': 0.1})
                # 新增代码行: 从params中获取斐波那契滚动窗口周期，默认55
                rolling_window_fib_period = params.get('rolling_window_fib_period', 55)
                required_cols = [
                    'volume', 'VOL_MA_21', 'main_force_flow_directionality',
                    'open', 'high', 'low', 'close',
                    'total_winner_rate', 'dominant_peak_solidity', 'VPA_EFFICIENCY',
                    'main_force_buy_execution_alpha', 'upward_impulse_strength',
                    'buy_order_book_clearing_rate', 'bid_side_liquidity',
                    'vwap_cross_up_intensity', 'opening_buy_strength',
                    'floating_chip_cleansing_efficiency',
                    'VPA_BUY_EFFICIENCY',
                    'deception_lure_long_intensity', 'wash_trade_buy_volume'
                ]
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    logger.warning(f"计算突破质量分(V2.7)失败，缺少必要列: {missing_cols}。")
                    return None
                # 维度一：能量输入 (0-1分)
                volume_ratio = df['volume'] / df['VOL_MA_21'].replace(0, np.nan)
                # 修改代码行: 使用 rolling_window_fib_period
                score_volume = (volume_ratio.rolling(rolling_window_fib_period).rank(pct=True) * 0.5 + \
                                df['buy_order_book_clearing_rate'].rolling(rolling_window_fib_period).rank(pct=True) * 0.3 + \
                                df['bid_side_liquidity'].rolling(rolling_window_fib_period).rank(pct=True) * 0.2).fillna(0.5)
                # 维度二：主导力量 (天然是-1到1分，映射到0-1)
                score_driver = ((df['main_force_buy_execution_alpha'].fillna(0) * 0.6 + \
                                 df['upward_impulse_strength'].fillna(0) * 0.4) + 1) / 2
                # 维度三：价格形态 (0-1分)
                price_range = (df['high'] - df['low']).replace(0, np.nan)
                raw_price_action = ((df['close'] - df['open']) / price_range).fillna(0)
                score_price_action = (raw_price_action.clip(-1, 1) * 0.5 + \
                                      df['vwap_cross_up_intensity'].fillna(0) * 0.3 + \
                                      df['opening_buy_strength'].fillna(0) * 0.2 + 1) / 2
                # 维度四：筹码结构 (0-1分)
                winner_rate_gain = df['total_winner_rate'].diff().fillna(0)
                chip_breakthrough_eff = winner_rate_gain * df['dominant_peak_solidity']
                # 修改代码行: 使用 rolling_window_fib_period
                score_chips = (chip_breakthrough_eff.rolling(rolling_window_fib_period).rank(pct=True) * 0.7 + \
                               df['floating_chip_cleansing_efficiency'].rolling(rolling_window_fib_period).rank(pct=True) * 0.3).fillna(0.5)
                # 维度五：攻击效率 (0-1分)
                # 修改代码行: 使用 rolling_window_fib_period
                score_efficiency = df['VPA_BUY_EFFICIENCY'].rolling(rolling_window_fib_period).rank(pct=True).fillna(0.5)
                # 维度六：纯度惩罚 (0-1分，越低越纯净，惩罚项)
                purity_penalty = (df['deception_lure_long_intensity'].fillna(0) * 0.7 + \
                                  df['wash_trade_buy_volume'].fillna(0) * 0.3).clip(0, 1)
                quality_score = (
                    score_volume * weights['volume'] +
                    score_driver * weights['driver'] +
                    score_price_action * weights['price_action'] +
                    score_chips * weights['chips'] +
                    score_efficiency * weights['efficiency'] -
                    purity_penalty * weights['purity_penalty']
                ).clip(0, 1)
                return pd.DataFrame({'breakout_quality_score': quality_score})
            return await asyncio.to_thread(_sync_calc)
        except Exception as e:
            logger.error(f"计算突破质量分(V2.7)时发生错误: {e}", exc_info=True)
            return None
    async def calculate_nolds_sample_entropy(self, df: pd.DataFrame, period: int, column: str, tolerance_ratio: float = 0.2) -> pd.Series:
        """
        【V1.7 · nolds样本熵集成版】计算样本熵 (Sample Entropy) 使用 nolds 库。
        - 核心修复: 鉴于 `pyentrp` 库在近似熵计算上的问题，现已切换为使用 `nolds` 库的 `sampen` 函数。
                  样本熵是近似熵的改进版本，通常更稳健。
        - 核心逻辑: 使用 `nolds.sampen` 计算滚动样本熵，衡量时间序列的复杂度和不可预测性。
        - 参数说明: `period` 在此函数中被视为滚动窗口大小。`emb_dim` (m) 固定为 2。
        - `tolerance_ratio`: 容忍度 `r` 的比例因子，`r = tolerance_ratio * std(window_data)`。
        """
        # 检查 nolds 库和 sampen 函数是否可用
        if nolds is None or not hasattr(nolds, 'sampen'):
            logger.error("样本熵计算失败：'nolds' 库未加载或不包含 'sampen' 函数。请确保已安装 'nolds'。")
            return pd.Series(np.nan, index=df.index)
        if column not in df.columns:
            logger.warning(f"样本熵计算失败: 列 '{column}' 不存在。")
            return pd.Series(np.nan, index=df.index)
        series_raw = df[column].astype(float)
        emb_dim = 2 # 嵌入维度 m，通常取 2
        min_samples_for_window = max(period, emb_dim + 1)
        if len(series_raw) < min_samples_for_window:
            logger.warning(f"样本熵计算失败: 序列 '{column}' 数据不足 (长度: {len(series_raw)}, 最小要求窗口: {min_samples_for_window})。")
            return pd.Series(np.nan, index=df.index)
        results = pd.Series(np.nan, index=df.index)
        for i in range(len(series_raw)):
            if i < min_samples_for_window - 1:
                continue
            window_data = series_raw.iloc[i - min_samples_for_window + 1 : i + 1].dropna().values
            if len(window_data) < min_samples_for_window:
                continue
            # 检查窗口数据是否为常数或变化极小
            if np.all(window_data == window_data[0]) or np.std(window_data) < 1e-9:
                results.iloc[i] = 0.0
                continue
            try:
                std_dev = np.std(window_data)
                r_tolerance = tolerance_ratio * std_dev
                if r_tolerance == 0:
                    results.iloc[i] = 0.0
                    continue
                # 调用 nolds.sampen 函数，并传递正确的参数
                samp_en = nolds.sampen(window_data, emb_dim=emb_dim, tolerance=r_tolerance)
                results.iloc[i] = samp_en
            except Exception as e:
                logger.error(f"样本熵(周期{period}, 列: {column})计算失败: {e} for series window ending at {series_raw.index[i]}. Window data (first 5): {window_data[:5]}...", exc_info=False)
                results.iloc[i] = np.nan
        return results.reindex(df.index)






