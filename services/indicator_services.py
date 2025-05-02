# services\indicator_services.py
import asyncio
from collections import defaultdict
import datetime
from functools import reduce
import json
import os
import warnings
import logging
from dao_manager.tushare_daos.indicator_dao import IndicatorDAO
import numpy as np
import pandas as pd
from django.utils import timezone
from typing import Any, List, Optional, Union
from django.db import models # 确保导入 models
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

from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from core.constants import TimeLevel

logger = logging.getLogger("services")

class IndicatorService:
    """
    技术指标计算服务 (使用 pandas-ta)
    """
    def __init__(self):
        self.indicator_dao = IndicatorDAO()
        self.stock_basic_dao = StockBasicInfoDao() # 用于获取 StockInfo 对象
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
        limit = needed_bars  # 增加一些 buffer
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

        if 'time_level' in df.columns:  # 假设你的 DataFrame 可能有 time_level 列
            try:
                df['time_level'] = df['time_level'].astype('category')
            except Exception as e:
                logger.warning(f"转换 time_level 列为 category 类型失败: {e}")
        # --------------------------------------------------------------------
        # 确保列名是小写，pandas-ta 推荐使用小写列名
        df.columns = [col.lower() for col in df.columns]
        # --- 添加：返回前，确保必要的列是 float 类型 ---
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        if 'amount' in df.columns: ohlcv_cols.append('amount')
        if 'turnover_rate' in df.columns: ohlcv_cols.append('turnover_rate')  # 新增换手率字段
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
                return None  # 无法继续
        default_tz = timezone.get_default_timezone()
        if df.index.tz is None:
            logger.warning(f"OHLCV 数据索引是 naive 的，将本地化为默认时区...")
            try:
                df.index = df.index.tz_localize(default_tz)
            except Exception as e_tz:
                logger.error(f"无法本地化 OHLCV 索引时区: {e_tz}")
                return None  # 无法继续
        elif df.index.tz != default_tz:
            # 如果已经是 aware 但时区不同，转换为默认时区
            df.index = df.index.tz_convert(default_tz)
        # --- 时区处理结束 ---
        return df

    async def prepare_strategy_dataframe(self, stock_code: str, params_file: str, needed_bars: int = None) -> Optional[pd.DataFrame]:
        """
        根据策略 JSON 配置文件准备包含基础数据和所有计算指标的 DataFrame。
        Args:
            stock_code (str): 股票代码，用于标识具体股票。
            params_file (str): 策略参数 JSON 文件的路径，包含时间框架和指标参数。
        Returns:
            Optional[pd.DataFrame]: 包含所有所需数据的 DataFrame，列名包含时间级别后缀（如 'RSI_12_15', 'close_60'）。
                                    如果数据准备失败，则返回 None。
        """
        if ta is None:
            logger.error(f"[{stock_code}] pandas_ta 未加载，无法准备策略数据。")
            return None
        # 1. 加载 JSON 参数文件
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
        # 2. 识别需求：时间级别和最大回看期
        all_time_levels = set()
        max_lookback = 0
        try:
            bs_params = params['base_scoring']  # 基础评分参数
            vc_params = params['volume_confirmation']  # 成交量确认参数
            dd_params = params['divergence_detection']  # 背离检测参数
            kpd_params = params['kline_pattern_detection']  # K线形态检测参数
            ta_params = params['trend_analysis']  # 趋势分析参数
            ia_params = params['indicator_analysis_params']  # 指标分析参数
            tr_params = params.get('trend_reversal_params', {})  # 趋势反转参数
            # 将所有需要的时间级别添加到集合中
            all_time_levels.update(bs_params['timeframes'])
            if vc_params['enabled']:
                all_time_levels.add(vc_params['tf'])
            if dd_params['enabled']:
                all_time_levels.add(dd_params['tf'])
            if kpd_params['enabled']:
                all_time_levels.add(kpd_params['tf'])
            # 确定分析所需的时间框架（优先级：背离 > K线形态 > 成交量 > 默认）
            analysis_tf = (dd_params.get('tf') if dd_params['enabled'] else
                        kpd_params.get('tf') if kpd_params['enabled'] else
                        vc_params.get('tf', bs_params['timeframes'][0]))
            all_time_levels.add(analysis_tf)
            # 为 T+0 策略添加 focus_timeframe（如果存在）
            t0_params = params.get('t_plus_0_signals', {})
            if t0_params.get('enabled', False):
                focus_timeframe = t0_params.get('focus_timeframe', '5')
                all_time_levels.add(focus_timeframe)
                logger.info(f"[{stock_code}] 为 T+0 策略添加时间级别: {focus_timeframe}")
            # 为换手率过滤添加时间框架（如果启用）
            turnover_filter = tr_params.get('turnover_filter', {})
            if turnover_filter.get('enabled', False):
                turnover_tf = turnover_filter.get('timeframe', analysis_tf)
                all_time_levels.add(turnover_tf)
                logger.info(f"[{stock_code}] 为换手率过滤添加时间级别: {turnover_tf}")
            # 计算最大回看期，考虑所有指标的参数，确保数据足够
            lookbacks = [
                bs_params.get('rsi_period', 0),  # RSI 周期
                bs_params.get('kdj_period_k', 0),  # KDJ K周期
                bs_params.get('boll_period', 0),  # 布林带周期
                bs_params.get('macd_slow', 0) + bs_params.get('macd_signal', 0),  # MACD 慢线+信号线周期
                bs_params.get('cci_period', 0),  # CCI 周期
                bs_params.get('mfi_period', 0),  # MFI 周期
                bs_params.get('roc_period', 0),  # ROC 周期
                bs_params.get('dmi_period', 0) * 3,  # DMI/ADX 需要更多数据
                vc_params.get('amount_ma_period', 0),  # 成交额均线周期
                vc_params.get('cmf_period', 0),  # CMF 周期
                vc_params.get('obv_ma_period', 0),  # OBV 均线周期
                dd_params.get('lookback', 0) * 2,  # 背离检测可能需要 2*lookback
                max(ta_params.get('ema_periods', [0])),  # 评分 EMA 周期
                ia_params.get('stoch_k', 0) + ia_params.get('stoch_d', 0) + ia_params.get('stoch_smooth_k', 0),  # 随机指标周期
                ia_params.get('volume_ma_period', 0),  # 成交量均线周期
                55  # SAR 默认回看期或固定值
            ]
            if needed_bars is not None:
                max_lookback = needed_bars
            else:
                max_lookback = max(lookbacks) + 100  # 增加 100 个 bar 作为缓冲
            logger.info(f"[{stock_code}] 需要的时间级别: {all_time_levels}, 最大回看期: {max_lookback}")
        except KeyError as e:
            logger.error(f"[{stock_code}] 参数文件 {params_file} 缺少键: {e}", exc_info=True)
            return None
        # 3. 并行获取基础 OHLCV 数据（开、高、低、收、成交量、换手率）
        ohlcv_tasks = {
            tf: self._get_ohlcv_data(stock_code, tf, max_lookback)
            for tf in all_time_levels
        }
        ohlcv_results = await asyncio.gather(*ohlcv_tasks.values())
        ohlcv_dfs = dict(zip(all_time_levels, ohlcv_results))
        # 输出每个时间级别获取到的数据量
        for tf, df in ohlcv_dfs.items():
            if df is not None and not df.empty:
                logger.info(f"[{stock_code}] 时间级别 {tf} 获取到 {len(df)} 条K线数据，时间范围: {df.index.min()} 至 {df.index.max()}")
            else:
                logger.warning(f"[{stock_code}] 时间级别 {tf} 未获取到数据或数据为空")
        # 检查是否有数据获取失败
        valid_ohlcv_dfs = {}
        for tf, df in ohlcv_dfs.items():
            if df is None or df.empty:
                logger.warning(f"[{stock_code}] 无法获取时间级别 {tf} 的 OHLCV 数据。")
            else:
                df = self.filter_to_period_points(df, tf)  # <--- 周期对齐
                # 重命名基础列，将 amount 重命名为 amount，并包含 turnover_rate
                rename_map = {}
                for col in df.columns:
                    if col in ['open', 'high', 'low', 'close', 'volume']:
                        rename_map[col] = f"{col}_{tf}"
                    elif col == 'amount':  # 特别处理 amount
                        rename_map[col] = f"amount_{tf}"  # 重命名为 amount_{tf}
                    elif col == 'turnover_rate':  # 新增换手率字段
                        rename_map[col] = f"turnover_rate_{tf}"
                    # 其他列名不变（如 stock_code, time_level）
                valid_ohlcv_dfs[tf] = df.rename(columns=rename_map)
        if not valid_ohlcv_dfs:
            logger.error(f"[{stock_code}] 无法获取任何有效的基础 OHLCV 数据。")
            return None
        # 4. 计算所有指标 - 使用并行任务
        calculated_indicators = defaultdict(list)  # {tf: [indicator_df1, indicator_df2, ...]}
        # 辅助函数：安全计算指标并存储结果
        async def _calculate_and_store_async(tf, indicator_name, calculation_func, *args, **kwargs):
            if tf in valid_ohlcv_dfs:
                base_df = ohlcv_dfs[tf]  # 使用原始未重命名的 DF 进行计算
                if base_df is None or base_df.empty:
                    logger.warning(f"[{stock_code}] 时间级别 {tf} 的基础数据为空，无法计算 {indicator_name}")
                    return None
                try:
                    # 确保基础数据是数值类型
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in base_df.columns:
                            base_df[col] = pd.to_numeric(base_df[col], errors='coerce')
                    base_df_clean = base_df.copy()
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in base_df_clean.columns:
                            base_df_clean[col].ffill(inplace=True)
                            base_df_clean[col].bfill(inplace=True)
                    if base_df_clean[['open', 'high', 'low', 'close', 'volume']].isnull().any(axis=1).all():
                        logger.warning(f"[{stock_code}] 清理 NaN 后时间级别 {tf} 数据仍为空，无法计算 {indicator_name}")
                        return None
                    result = calculation_func(base_df_clean, *args, **kwargs)
                    if result is not None and not result.empty:
                        result = self.filter_to_period_points(result, tf)  # <--- 周期对齐
                        # 统一为所有结果列添加时间后缀
                        result_renamed = result.rename(columns=lambda x: f"{x}_{tf}")
                        return (tf, result_renamed)
                    else:
                        logger.warning(f"[{stock_code}] 计算 {indicator_name} for 时间级别 {tf} 结果为空或失败")
                        return None
                except Exception as e:
                    logger.error(f"[{stock_code}] 计算指标 {indicator_name} (时间 {tf}) 时出错: {e}", exc_info=True)
                    return None
            return None
        # 创建并行任务列表
        indicator_tasks = []
        for indi_key in bs_params.get('score_indicators', []):
            for tf in bs_params.get('timeframes', []):
                if indi_key == 'macd':
                    indicator_tasks.append(_calculate_and_store_async(tf, 'MACD', self.calculate_macd,
                                                                    period_fast=bs_params['macd_fast'],
                                                                    period_slow=bs_params['macd_slow'],
                                                                    signal_period=bs_params['macd_signal']))
                elif indi_key == 'rsi':
                    indicator_tasks.append(_calculate_and_store_async(tf, 'RSI', self.calculate_rsi, period=bs_params['rsi_period']))
                elif indi_key == 'kdj':
                    indicator_tasks.append(_calculate_and_store_async(tf, 'KDJ', self.calculate_kdj,
                                                                    period=bs_params['kdj_period_k'],
                                                                    signal_period=bs_params['kdj_period_d'],
                                                                    smooth_k_period=bs_params['kdj_period_j']))
                elif indi_key == 'boll':
                    indicator_tasks.append(_calculate_and_store_async(tf, 'BOLL', self.calculate_boll, period=bs_params['boll_period'], std_dev=bs_params['boll_std_dev']))
                elif indi_key == 'cci':
                    indicator_tasks.append(_calculate_and_store_async(tf, 'CCI', self.calculate_cci, period=bs_params['cci_period']))
                elif indi_key == 'mfi':
                    indicator_tasks.append(_calculate_and_store_async(tf, 'MFI', self.calculate_mfi, period=bs_params['mfi_period']))
                elif indi_key == 'roc':
                    indicator_tasks.append(_calculate_and_store_async(tf, 'ROC', self.calculate_roc, period=bs_params['roc_period']))
                elif indi_key == 'dmi':
                    indicator_tasks.append(_calculate_and_store_async(tf, 'DMI', self.calculate_dmi, period=bs_params['dmi_period']))
                elif indi_key == 'sar':
                    indicator_tasks.append(_calculate_and_store_async(tf, 'SAR', self.calculate_sar, af=bs_params['sar_step'], max_af=bs_params['sar_max']))
        # 计算成交量确认指标
        if vc_params['enabled']:
            tf = vc_params['tf']
            indicator_tasks.append(_calculate_and_store_async(tf, 'AMT_MA', self.calculate_amount_ma, period=vc_params['amount_ma_period']))
            indicator_tasks.append(_calculate_and_store_async(tf, 'CMF', self.calculate_cmf, period=vc_params['cmf_period']))
            obv_timeframes = bs_params.get('timeframes', [])
            for tf in obv_timeframes:
                indicator_tasks.append(_calculate_and_store_async(tf, 'OBV', self.calculate_obv))
        # 计算分析所需的指标 - 对所有时间级别计算 STOCH 和 VOL_MA
        for tf in bs_params.get('timeframes', []):
            indicator_tasks.append(_calculate_and_store_async(tf, 'STOCH', self.calculate_stoch,
                                                            k_period=ia_params['stoch_k'],
                                                            d_period=ia_params['stoch_d'],
                                                            smooth_k_period=ia_params['stoch_smooth_k']))
            vol_ma_col_name = f'VOL_MA_{ia_params["volume_ma_period"]}_{tf}'
            vol_ma_exists = any(vol_ma_col_name in df.columns for df in calculated_indicators.get(tf, []))
            if not vol_ma_exists:
                indicator_tasks.append(_calculate_and_store_async(tf, 'VOL_MA', lambda df: df.ta.sma(close='volume', length=ia_params['volume_ma_period']).to_frame(name=f'VOL_MA_{ia_params["volume_ma_period"]}')))
        # 计算布林带（如果未计算）
        bb_tf = analysis_tf
        bb_col_name = f'BB_UPPER_{bs_params["boll_period"]}_{bb_tf}'
        bb_exists = any(bb_col_name in df.columns for df in calculated_indicators.get(bb_tf, []))
        if not bb_exists:
            indicator_tasks.append(_calculate_and_store_async(bb_tf, 'BOLL', self.calculate_boll, period=bs_params['boll_period'], std_dev=bs_params['boll_std_dev']))
        # 计算 VWAP（无论 T+0 策略是否启用，确保在所有时间级别上计算）
        for tf in bs_params.get('timeframes', []):
            indicator_tasks.append(_calculate_and_store_async(tf, 'VWAP', self.calculate_vwap))
        # 如果 T+0 策略启用，额外确保 focus_timeframe 被覆盖
        if params.get('t_plus_0_signals', {}).get('enabled'):
            vwap_tf = analysis_tf
            indicator_tasks.append(_calculate_and_store_async(vwap_tf, 'VWAP', self.calculate_vwap))
        # 并行执行所有指标计算任务
        results = await asyncio.gather(*indicator_tasks, return_exceptions=True)
        for result in results:
            if result and isinstance(result, tuple):
                tf, indicator_df = result
                calculated_indicators[tf].append(indicator_df)
        # 后处理 OBV 均线（需要在 OBV 计算后）
        if vc_params['enabled']:
            obv_timeframes = bs_params.get('timeframes', [])
            for tf in obv_timeframes:
                if tf in calculated_indicators:
                    obv_df = next((df for df in calculated_indicators[tf] if f'OBV_{tf}' in df.columns), None)
                    if obv_df is not None:
                        obv_series = obv_df[f'OBV_{tf}']
                        if not obv_series.isnull().all():
                            obv_ma_period = vc_params.get('obv_ma_period', 10)
                            obv_ma_series = ta.sma(obv_series, length=obv_ma_period)
                            if obv_ma_series is not None:
                                obv_ma_df = obv_ma_series.to_frame(name=f'OBV_MA_{obv_ma_period}_{tf}')
                                calculated_indicators[tf].append(obv_ma_df)
        # 后处理布林带列名修正
        for tf in calculated_indicators:
            for df in calculated_indicators[tf]:
                if any(col.startswith('BB_UPPER_') for col in df.columns):
                    period = bs_params["boll_period"]
                    df.rename(columns={
                        f'BB_UPPER_{period}': f'BB_UPPER_{period}_{tf}',
                        f'BB_MIDDLE_{period}': f'BB_MIDDLE_{period}_{tf}',
                        f'BB_LOWER_{period}': f'BB_LOWER_{period}_{tf}'
                    }, inplace=True)
        # 5. 合并所有数据到一个 DataFrame - 使用 pd.concat 提高性能，不以5分钟为基准对齐
        all_dfs = []
        for tf in valid_ohlcv_dfs:
            # 添加基础 OHLCV 数据
            all_dfs.append(valid_ohlcv_dfs[tf][[col for col in valid_ohlcv_dfs[tf].columns if col.endswith(f'_{tf}')]])
            # 添加计算的指标数据
            for indicator_df in calculated_indicators.get(tf, []):
                all_dfs.append(indicator_df)
        if not all_dfs:
            logger.error(f"[{stock_code}] 没有可用的数据帧进行合并。")
            return None
        # 使用 pd.concat 合并所有 DataFrame，按索引外连接对齐，并处理重复列名
        try:
            combined_df = pd.concat(all_dfs, axis=1, join='outer', copy=False)
            # 处理重复列名，保留第一个出现的列
            combined_df = combined_df.loc[:, ~combined_df.columns.duplicated(keep='first')]
            logger.info(f"[{stock_code}] 成功合并所有数据帧，形状: {combined_df.shape}")
            # 对合并后的数据进行填充，确保数据完整性
            combined_df.ffill(inplace=True)
            combined_df.bfill(inplace=True)
            # =========================
            self._log_dataframe_missing(combined_df, stock_code)
            # =========================
            logger.info(f"[{stock_code}] 合并后数据填充完成，最终形状: {combined_df.shape}")
            # 彻底消除None，强制所有列为float
            for col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
            return combined_df
        except Exception as e:
            logger.error(f"[{stock_code}] 合并数据帧时出错: {e}", exc_info=True)
            return None

    # --- 周期对齐函数 ---
    def filter_to_period_points(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """
        只保留周期对齐的时间点，自动适配K线收盘分钟。
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        if tf.isdigit() and int(tf) < 60:
            period = int(tf)
            # 取所有分钟的模period的分布，选出现次数最多的那个余数
            mod_counts = df.index.minute % period
            most_common_mod = mod_counts.value_counts().idxmax()
            mask = (df.index.minute % period == most_common_mod) & (df.index.second == 0)
            return df[mask]
        else:
            return df

    # --- 检查缺失项目并记录 ---
    def _log_dataframe_missing(self, df: pd.DataFrame, stock_code: str):
        missing_count = df.isna().sum()
        missing_ratio = (df.isna().mean() * 100).round(2)
        logger.warning(f"[{stock_code}] 合并后各列缺失数量: {missing_count.to_dict()}")
        logger.warning(f"[{stock_code}] 合并后各列缺失比例(%): {missing_ratio.to_dict()}")
        all_nan_rows = df.isna().all(axis=1).sum()
        if all_nan_rows > 0:
            logger.warning(f"[{stock_code}] 合并后DataFrame存在{all_nan_rows}行全部为NaN的数据！")
        # 突出关键列的缺失情况
        key_cols = [col for col in df.columns if any(k in col for k in ['open', 'high', 'low', 'close', 'volume'])]
        if key_cols:
            key_missing_count = missing_count[key_cols].to_dict()
            key_missing_ratio = missing_ratio[key_cols].to_dict()
            logger.warning(f"[{stock_code}] 关键列缺失数量: {key_missing_count}")
            logger.warning(f"[{stock_code}] 关键列缺失比例(%): {key_missing_ratio}")
        top_missing = missing_ratio.sort_values(ascending=False).head(5)
        logger.warning(f"[{stock_code}] 缺失比例最高的5列: {top_missing.to_dict()}")


    def calculate_atr(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 ATR。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period (int): ATR 计算周期。
        Returns:
            Optional[pd.DataFrame]: 包含 "ATR_period" 列的 DataFrame，如果计算失败则返回 None。
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
        """
        计算布林带 (标准周期)
        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period (int): 布林带计算周期。
            std_dev (float): 标准差倍数。
        Returns:
            Optional[pd.DataFrame]: 包含 'BB_UPPER_period', 'BB_MIDDLE_period', 'BB_LOWER_period'
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < period: return None
        try:
            bbands_df = ohlc.ta.bbands(length=period, std=std_dev)
            if bbands_df is None or bbands_df.empty:
                return None
            # 重命名列以匹配 finta 的输出，并包含 period 信息
            rename_map = {
                f'BBU_{period}_{std_dev}': f'BB_UPPER_{period}',
                f'BBM_{period}_{std_dev}': f'BB_MIDDLE_{period}',
                f'BBL_{period}_{std_dev}': f'BB_LOWER_{period}',
            }
            required_cols = list(rename_map.keys())
            actual_cols = {col: rename_map[col] for col in required_cols if col in bbands_df.columns}
            if len(actual_cols) != 3:
                logger.warning(f"pandas-ta bbands 未返回所有预期列 for period={period}, std={std_dev}. 返回: {bbands_df.columns.tolist()}")
                potential_map = {'BBU': f'BB_UPPER_{period}', 'BBM': f'BB_MIDDLE_{period}', 'BBL': f'BB_LOWER_{period}'}
                common_cols = {f'{k}_{period}_{std_dev}': v for k, v in potential_map.items() if f'{k}_{period}_{std_dev}' in bbands_df.columns}
                if len(common_cols) == 3:
                    actual_cols = common_cols
                else:
                    logger.error(f"无法从 pandas-ta bbands 结果中提取所需列。")
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
            pdi_col_ta = f'DMP_{period}'
            mdi_col_ta = f'DMN_{period}'
            adx_col_ta = f'ADX_{period}'
            pdi_col_strat = f'+DI_{period}'
            mdi_col_strat = f'-DI_{period}'
            adx_col_strat = f'ADX_{period}' # 保持 ADX 名称
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
        """
        计算一目均衡表示线 (Ichimoku)。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
        Returns:
            Optional[pd.DataFrame]: 包含 'TENKAN', 'KIJUN', 'CHIKOU', 'SENKOU A', 'SENKOU B' 列的 DataFrame。
        """
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
        
    def calculate_amount_ma(self, ohlc: pd.DataFrame, period: int, tf: str = "") -> Optional[pd.DataFrame]:
        """
        计算指定周期的成交额 SMA，返回列名为 AMT_MA_周期_时间框，满足策略列名需求。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据，必须包含 'amount' 列。
            period (int): SMA 计算周期。
            tf (str): 时间框后缀，例如 '15', 用于生成列名。
        Returns:
            Optional[pd.DataFrame]: 计算结果DataFrame，列名如 'AMT_MA_15_15'。
                                计算失败返回 None。
        """
        if ta is None:
            logger.error("pandas-ta 未加载，无法计算成交额均线")
            return None
        if ohlc is None or ohlc.empty:
            logger.error("输入的 OHLCV 数据为空，无法计算成交额均线")
            return None
        if 'amount' not in ohlc.columns:
            logger.error("计算 Amount MA 需要数据包含 'amount' 列")
            return None
        if len(ohlc) < period:
            logger.warning(f"数据长度 {len(ohlc)} 小于所需周期 {period}，成交额均线计算返回 None")
            return None
        try:
            # pandas_ta 用 sma 计算成交额均线，输入列名为 'amount'
            amt_ma_series = ohlc.ta.sma(close='amount', length=period)
            if amt_ma_series is None or amt_ma_series.empty:
                logger.warning("成交额均线计算结果为空")
                return None

            # 构建列名，格式为 AMT_MA_周期_时间框
            col_name = f'AMT_MA_{period}'
            if tf:
                col_name = f'AMT_MA_{period}_{tf}'

            # 转换为 DataFrame，返回
            result_df = amt_ma_series.to_frame(name=col_name)
            return result_df

        except Exception as e:
            logger.error(f"计算成交额均线 AMT_MA(period={period}, tf={tf}) 失败: {e}", exc_info=True)
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
        """
        计算 OBV。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
        Returns:
            Optional[pd.DataFrame]: 包含 'OBV' 列的 DataFrame，如果计算失败则返回 None。
        """
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
            ohlc (pd.DataFrame): OHLCV 数据 (必须包含 'amount' 列)。
            period (int): ROC 计算周期。
        Returns:
            Optional[pd.DataFrame]: 包含 'AROC_period' 列的 DataFrame，如果计算失败则返回 None。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or 'amount' not in ohlc.columns:
             logger.error("计算 Amount ROC 需要 'amount' 列")
             return None
        if len(ohlc) < period + 1: return None
        try:
            aroc_series = ohlc.ta.roc(close='amount', length=period)
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
            # 斐波那契枢轴点
            diff = prev_high - prev_low
            results['F_R1'] = PP + 0.382 * diff
            results['F_R2'] = PP + 0.618 * diff
            results['F_R3'] = PP + 1.000 * diff
            results['F_S1'] = PP - 0.382 * diff
            results['F_S2'] = PP - 0.618 * diff
            results['F_S3'] = PP - 1.000 * diff
            # 删除第一行（无法计算）
            results = results.iloc[1:]
            return results
        except Exception as e:
            logger.error(f"计算 Pivot Points 失败: {e}", exc_info=True)
            return None










