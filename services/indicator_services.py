import asyncio
from collections import defaultdict
from functools import reduce
import json
import os
import warnings
import logging
from dao_manager.tushare_daos.indicator_dao import IndicatorDAO
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
        if 'turnover' in df.columns: ohlcv_cols.append('turnover')
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

    async def prepare_strategy_dataframe(self, stock_code: str, params_file: str) -> Optional[pd.DataFrame]:
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
            max_lookback = max(lookbacks) + 50  # 增加 50 个 bar 作为缓冲
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

        # 检查是否有数据获取失败
        valid_ohlcv_dfs = {}
        for tf, df in ohlcv_dfs.items():
            if df is None or df.empty:
                logger.warning(f"[{stock_code}] 无法获取时间级别 {tf} 的 OHLCV 数据。")
            else:
                # 重命名基础列，将 turnover 重命名为 amount，并包含 turnover_rate
                rename_map = {}
                for col in df.columns:
                    if col in ['open', 'high', 'low', 'close', 'volume']:
                        rename_map[col] = f"{col}_{tf}"
                    elif col == 'turnover':  # 特别处理 turnover
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
                    base_df_clean = base_df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
                    if base_df_clean.empty:
                        logger.warning(f"[{stock_code}] 清理 NaN 后时间级别 {tf} 数据为空，无法计算 {indicator_name}")
                        return None
                    result = calculation_func(base_df_clean, *args, **kwargs)
                    if result is not None and not result.empty:
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
        # 5. 合并所有数据到一个 DataFrame - 使用 pd.concat 提高性能
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
        # 使用 pd.concat 合并所有 DataFrame，按索引对齐，并处理重复列名
        try:
            combined_df = pd.concat(all_dfs, axis=1, join='outer', copy=False)
            # 处理重复列名，保留第一个出现的列
            combined_df = combined_df.loc[:, ~combined_df.columns.duplicated(keep='first')]
            logger.info(f"[{stock_code}] 成功合并所有数据帧，形状: {combined_df.shape}")
            return combined_df
        except Exception as e:
            logger.error(f"[{stock_code}] 合并数据帧时出错: {e}", exc_info=True)
            return None
    
    # --- 新增方法：准备基础 OHLCV 数据 ---
    async def prepare_strategy_basic_dataframe(self, stock_code: str, timeframes: List[str], limit_per_tf: int = 1200) -> Optional[pd.DataFrame]:
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












