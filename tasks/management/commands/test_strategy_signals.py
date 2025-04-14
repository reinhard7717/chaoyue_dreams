# tasks/management/commands/test_strategy_signals.py

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from django.core.management.base import BaseCommand
import logging
from django.utils import timezone # 用于处理时区

# --- 导入时区处理库 ---
try:
    import tzlocal
    from zoneinfo import ZoneInfo
except ImportError:
    tzlocal = None
    ZoneInfo = None
    print("警告：无法导入 'tzlocal' 或 'zoneinfo'。")

# --- 导入 pandas_ta ---
try:
    import pandas_ta as ta
except ImportError:
    ta = None
    print("警告：无法导入 'pandas_ta'。")

# --- 导入项目模块 ---
from dao_manager.daos.stock_basic_dao import StockBasicDAO
from services.indicator_services import IndicatorService # 依赖这个服务来准备数据
from strategies.macd_rsi_kdj_boll_strategy import MacdRsiKdjBollEnhancedStrategy
# 不再需要单独导入 IndicatorDAO 来获取 VWAP
from django.core.cache import cache

# --- 导入分析函数 ---
# 假设 analyze_score_trend 函数定义在同一个文件或已正确导入
# from .trend_analyzer import analyze_score_trend # 如果在单独文件

logger = logging.getLogger(__name__)

# --- analyze_score_trend 函数 (使用 VWAP 版本) ---
# (这里放置之前生成的 analyze_score_trend 函数代码, 函数签名不变)
def analyze_score_trend(
    stock_code: str,
    score_price_vwap_df: pd.DataFrame, # 输入：包含 'score', 'close', 'vwap' 列
    t0_params: Optional[Dict[str, Any]] = None
) -> Optional[pd.DataFrame]:
    """
    细化分析策略评分和价格的趋势，并使用 VWAP 提供日内 T+0 交易信号提示。
    (函数体省略，使用上一轮生成的代码)
    """
    # --- 参数处理 ---
    default_t0_params = {
        'enabled': True,
        'buy_dev_threshold': -0.003, # 价格低于 VWAP 0.3%
        'sell_dev_threshold': 0.005 # 价格高于 VWAP 0.5%
    }
    if t0_params is None: t0_params = default_t0_params
    else: t0_params = {**default_t0_params, **t0_params}

    # --- 输入数据验证 ---
    if ta is None: logger.warning(f"[{stock_code}] pandas_ta 未安装..."); return None
    if score_price_vwap_df is None or score_price_vwap_df.empty: logger.warning(f"[{stock_code}] 输入 DF 为空..."); return None
    required_cols = ['score', 'close']
    if t0_params['enabled']: required_cols.append('vwap')
    missing_cols = [col for col in required_cols if col not in score_price_vwap_df.columns]
    if missing_cols:
        logger.warning(f"[{stock_code}] 输入 DF 缺少列: {missing_cols}。")
        if 'vwap' in missing_cols and t0_params['enabled']:
            logger.warning(f"[{stock_code}] 缺少 'vwap' 列，禁用 T+0。")
            t0_params['enabled'] = False
            if 'score' in missing_cols or 'close' in missing_cols: return None
        elif 'score' in missing_cols or 'close' in missing_cols: return None
    if not isinstance(score_price_vwap_df.index, pd.DatetimeIndex): logger.warning(f"[{stock_code}] 索引非 DatetimeIndex..."); return None
    min_required_data = 55
    if len(score_price_vwap_df) < min_required_data: logger.warning(f"[{stock_code}] 数据点不足 {min_required_data}..."); return None

    logger.info(f"[{stock_code}] 开始分析 (T+0 基于 VWAP)...")
    analysis_df = score_price_vwap_df.copy()

    # 1. 标准化时区
    try:
        if ZoneInfo:
            target_tz = ZoneInfo("Asia/Shanghai")
            if analysis_df.index.tz is None:
                try: analysis_df.index = analysis_df.index.tz_localize(target_tz)
                except Exception: analysis_df.index = analysis_df.index.tz_localize('UTC').tz_convert(target_tz)
            elif analysis_df.index.tz != target_tz: analysis_df.index = analysis_df.index.tz_convert(target_tz)
        else: logger.warning(f"[{stock_code}] zoneinfo 不可用...")
    except Exception as e: logger.error(f"[{stock_code}] 处理时区出错: {e}", exc_info=True); return None

    # 2. 计算评分 EMA
    fib_periods = [5, 13, 21, 55]
    try:
        for period in fib_periods: analysis_df[f'ema_score_{period}'] = ta.ema(analysis_df['score'], length=period)
    except Exception as e: logger.error(f"[{stock_code}] 计算评分 EMA 出错: {e}", exc_info=True); return analysis_df

    # 3. 计算评分趋势排列信号
    signal_5_13 = np.where(analysis_df['ema_score_5'] > analysis_df['ema_score_13'], 1, np.where(analysis_df['ema_score_5'] < analysis_df['ema_score_13'], -1, 0))
    signal_13_21 = np.where(analysis_df['ema_score_13'] > analysis_df['ema_score_21'], 1, np.where(analysis_df['ema_score_13'] < analysis_df['ema_score_21'], -1, 0))
    signal_21_55 = np.where(analysis_df['ema_score_21'] > analysis_df['ema_score_55'], 1, np.where(analysis_df['ema_score_21'] < analysis_df['ema_score_55'], -1, 0))
    analysis_df['alignment_signal'] = signal_5_13 + signal_13_21 + signal_21_55
    analysis_df.loc[analysis_df[[f'ema_score_{p}' for p in fib_periods]].isna().any(axis=1), 'alignment_signal'] = np.nan

    # 4. 计算评分趋势强度
    analysis_df['ema_strength_13_55'] = analysis_df['ema_score_13'] - analysis_df['ema_score_55']
    # 5. 计算评分动能
    analysis_df['score_momentum'] = analysis_df['score'].diff()
    # 6. 计算评分波动性
    volatility_window = 10
    analysis_df['score_volatility'] = analysis_df['score'].rolling(window=volatility_window).std()

    # --- T+0 相关计算 (基于 VWAP) ---
    analysis_df['t0_signal'] = 0
    if t0_params['enabled']:
        logger.debug(f"[{stock_code}] 计算 T+0 指标 (基于 VWAP)...")
        buy_dev_threshold = t0_params['buy_dev_threshold']
        sell_dev_threshold = t0_params['sell_dev_threshold']
        if 'vwap' in analysis_df.columns:
            analysis_df['price_vwap_deviation'] = np.where(
                analysis_df['vwap'].isna() | (analysis_df['vwap'] == 0), np.nan,
                (analysis_df['close'] - analysis_df['vwap']) / analysis_df['vwap']
            )
        else: logger.warning(f"[{stock_code}] 'vwap' 列不存在，禁用 T+0。"); t0_params['enabled'] = False

        if t0_params['enabled'] and 'price_vwap_deviation' in analysis_df.columns:
            is_score_uptrend = analysis_df['alignment_signal'] >= 1
            is_score_downtrend = analysis_df['alignment_signal'] <= -1
            is_price_below_vwap = analysis_df['price_vwap_deviation'] < buy_dev_threshold
            is_price_above_vwap = analysis_df['price_vwap_deviation'] > sell_dev_threshold
            analysis_df.loc[is_score_uptrend & is_price_below_vwap, 't0_signal'] = 1
            analysis_df.loc[is_score_downtrend & is_price_above_vwap, 't0_signal'] = -1
            logger.debug(f"[{stock_code}] T+0 信号 (基于 VWAP) 计算完成。")
        elif t0_params['enabled']: logger.warning(f"[{stock_code}] 未计算 'price_vwap_deviation'..."); t0_params['enabled'] = False

    # 7. 综合分析与输出
    if not analysis_df.empty:
        latest_data = analysis_df.iloc[-1]
        latest_time = analysis_df.index[-1]
        summary = f"[{stock_code}] 最新评分与价格趋势分析 ({latest_time.strftime('%Y-%m-%d %H:%M:%S %Z')}):\n"
        summary += f"  - 最新评分: {latest_data['score']:.2f}, 最新价格: {latest_data['close']:.2f}, 最新VWAP: {latest_data.get('vwap', 'N/A'):.2f}\n"
        summary += f"  - 评分 EMA: 5={latest_data['ema_score_5']:.2f}, 13={latest_data['ema_score_13']:.2f}, 21={latest_data['ema_score_21']:.2f}, 55={latest_data['ema_score_55']:.2f}\n"
        alignment = latest_data['alignment_signal']
        if pd.isna(alignment): summary += "  - 评分趋势排列: 信号不足 (NaN)\n"
        elif alignment == 3: summary += "  - 评分趋势排列: 完全多头 (+3)\n"
        elif alignment == -3: summary += "  - 评分趋势排列: 完全空头 (-3)\n"
        elif alignment > 0: summary += f"  - 评分趋势排列: 偏多头 ({int(alignment)})\n"
        elif alignment < 0: summary += f"  - 评分趋势排列: 偏空头 ({int(alignment)})\n"
        else: summary += "  - 评分趋势排列: 混合/粘合 (0)\n"
        strength = latest_data['ema_strength_13_55']
        if pd.isna(strength): summary += "  - 评分趋势强度 (EMA13-55): NaN\n"
        else: summary += f"  - 评分趋势强度 (EMA13-55): {strength:.2f}\n"
        # d. 评分动能 (增加方向指示)
        momentum = latest_data['score_momentum']
        if pd.isna(momentum):
            summary += "  - 评分动能 (单期变化): 无法计算 (NaN)\n"
        else:
            summary += f"  - 评分动能 (单期变化): {momentum:.2f} " # 显示数值
            if momentum > 0:
                summary += "**(评分上升)**\n" # 明确上升
            elif momentum < 0:
                summary += "**(评分下降)**\n" # 明确下降
            else:
                summary += "(评分持平)\n" # 持平状态

        # e. 信号稳定性 (检查最近 3 期 alignment_signal)
        summary += "  - 信号稳定性 (近3期): "
        if len(analysis_df) >= 3:
            # 获取最近三期的排列信号，并过滤掉 NaN 值
            recent_alignment_raw = analysis_df['alignment_signal'].iloc[-3:].tolist()
            recent_alignment = [a for a in recent_alignment_raw if not pd.isna(a)] # 过滤 NaN

            if not recent_alignment: # 如果过滤后为空（例如最近三期都是 NaN）
                summary += "信号不足或无法判断\n"
            elif len(set(recent_alignment)) == 1: # 如果信号一致
                last_valid_signal = recent_alignment[-1]
                if last_valid_signal == 3:
                    summary += "连续完全多头排列\n"
                elif last_valid_signal == -3:
                    summary += "连续完全空头排列\n"
                elif last_valid_signal > 0:
                    summary += f"连续偏多头排列 ({int(last_valid_signal)})\n"
                elif last_valid_signal < 0:
                    summary += f"连续偏空头排列 ({int(last_valid_signal)})\n"
                else: # all zeros
                    summary += "连续混合/粘合状态\n"
            else: # 如果信号不一致
                summary += "排列信号波动\n"
        else: # 数据不足三期
            summary += "数据不足 (<3期)\n"

        # f. 评分波动性 (基于 10 期滚动标准差)
        volatility = latest_data['score_volatility']
        summary += f"  - 评分波动性 ({volatility_window}期 std): " # volatility_window 是之前定义的
        if pd.isna(volatility):
            summary += "无法计算 (NaN)\n"
        else:
            summary += f"{volatility:.2f} " # 显示数值
            # 计算波动性的分位数，以提供相对高低的判断依据
            try:
                # 忽略 NaN 值计算分位数，确保有足够非 NaN 值
                valid_volatility = analysis_df['score_volatility'].dropna()
                if len(valid_volatility) > volatility_window: # 需要足够数据才有意义
                    q25 = valid_volatility.quantile(0.25)
                    q75 = valid_volatility.quantile(0.75)
                    if volatility > q75:
                        summary += "**(近期波动较大)**\n" # 高于 75% 分位数
                    elif volatility < q25:
                        summary += "(近期波动较小)\n" # 低于 25% 分位数
                    else:
                        summary += "(近期波动适中)\n" # 介于 25% 和 75% 之间
                else:
                    summary += "(数据不足无法判断相对水平)\n"
            except Exception as e_quantile:
                 logger.warning(f"[{stock_code}] 计算波动性分位数时出错: {e_quantile}")
                 summary += "(无法判断相对水平)\n" # 计算分位数出错

        if t0_params['enabled']:
            summary += f"--- 日内 T+0 交易信号 (基于价格与 VWAP 偏离度) ---\n"
            price_dev = latest_data.get('price_vwap_deviation', np.nan)
            t0_sig = latest_data.get('t0_signal', 0)
            if pd.isna(price_dev): summary += "  - 价格相对 VWAP 偏离度: 无法计算 (NaN)\n"; summary += "  - T+0 信号: 无\n"
            else:
                summary += f"  - 价格相对 VWAP 偏离度: {price_dev:.2%}\n"
                if t0_sig == 1: summary += f"  - T+0 信号: **潜在买入点** (评分趋势向好且价格低于 VWAP 阈值 {t0_params['buy_dev_threshold']:.2%})\n"
                elif t0_sig == -1: summary += f"  - T+0 信号: **潜在卖出点** (评分趋势向差且价格高于 VWAP 阈值 {t0_params['sell_dev_threshold']:.2%})\n"
                else:
                    summary += "  - T+0 信号: 无或观望\n"
                    if not pd.isna(alignment):
                        if alignment >= 1 and price_dev >= t0_params['buy_dev_threshold']: summary += "      (原因: 评分趋势向好，但价格未低于 VWAP 买入阈值)\n"
                        elif alignment <= -1 and price_dev <= t0_params['sell_dev_threshold']: summary += "      (原因: 评分趋势向差，但价格未高于 VWAP 卖出阈值)\n"
                        elif alignment == 0: summary += "      (原因: 评分趋势不明朗)\n"
        else: summary += "--- 日内 T+0 交易信号: 未启用或无法计算 ---\n"

        print("\n" + "="*30 + " 评分与价格趋势分析摘要 " + "="*30); print(summary); print("="*78)
        print(f"\n[{stock_code}] 详细趋势分析数据 (最后10条，时间：Asia/Shanghai):")
        display_cols = ['score', 'close']
        if 'vwap' in analysis_df.columns: display_cols.append('vwap')
        display_cols.extend([f'ema_score_{p}' for p in fib_periods])
        display_cols.extend(['alignment_signal', 'ema_strength_13_55', 'score_momentum', 'score_volatility'])
        if t0_params['enabled']:
            if 'price_vwap_deviation' in analysis_df.columns: display_cols.append('price_vwap_deviation')
            if 't0_signal' in analysis_df.columns: display_cols.append('t0_signal')
        display_cols = [col for col in display_cols if col in analysis_df.columns]
        if display_cols:
            with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', 1200, 'display.float_format', '{:.2f}'.format):
                 formatters = {'price_vwap_deviation': '{:.2%}'.format} if 'price_vwap_deviation' in display_cols else {}
                 print(analysis_df[display_cols].tail(10).to_string(formatters=formatters))
        else: logger.warning(f"[{stock_code}] 无法显示详细趋势分析数据...")
    else: logger.warning(f"[{stock_code}] 分析结果 DF 为空..."); return None

    # 8. (可选) 保存结果到 Redis
    try:
        cache_key = f"strategy_score_trend:{stock_code}"
        analysis_df_json = analysis_df.reset_index()
        analysis_df_json['trade_time'] = analysis_df_json['trade_time'].dt.strftime('%Y-%m-%dT%H:%M:%S%z')
        analysis_df_serializable = analysis_df_json.replace({np.nan: None})
        cache.set(cache_key, analysis_df_serializable.to_json(orient='records', date_format='iso'), timeout=60 * 60)
        logger.info(f"[{stock_code}] 分析结果已缓存至 Redis (Key: {cache_key})。")
    except Exception as e: logger.error(f"[{stock_code}] 缓存分析结果至 Redis 出错: {e}", exc_info=True)

    return analysis_df
# --- 结束 analyze_score_trend 函数 ---


# --- 完整的 test_strategy_scores 异步函数 (基于 prepare_strategy_dataframe 已包含所有数据) ---
async def test_strategy_scores(stock_code: str, time_level_for_analysis: str = '5'):
    """
    测试指定股票代码的策略评分生成过程，并进行趋势和 T+0 分析。
    假设 prepare_strategy_dataframe 已返回包含所有所需指标 (含 EMA, VWAP) 的 DataFrame。

    Args:
        stock_code (str): 股票代码。
        time_level_for_analysis (str): 用于 T+0 分析和选取 close/vwap 列的时间周期 (例如 '1', '5', '15')。
                                       策略评分本身可能依赖其他周期。
    """
    # --- 获取本地时区 ---
    local_tz = None
    local_tz_name = "系统默认"
    if tzlocal and ZoneInfo:
        try:
            local_tz = tzlocal.get_localzone()
            local_tz_name = str(local_tz)
            logger.info(f"检测到本地时区: {local_tz_name}")
        except Exception as tz_e:
            logger.warning(f"获取本地时区时出错: {tz_e}. 时间将不会转换。")
            local_tz = None
    else:
        logger.warning("'tzlocal' 或 'zoneinfo' 不可用，时间将不会转换为本地时区。")

    # 1. 初始化服务和 DAO 实例
    indicator_service = IndicatorService() # 主要用于调用 prepare_strategy_dataframe
    stock_basic_dao = StockBasicDAO()
    # 不再需要 IndicatorDAO 来单独获取 VWAP

    # 2. 定义策略参数 (需要包含 EMA 周期参数，假设策略内部会用到)
    strategy_params: Dict[str, Any] = {
        # ... (其他指标参数如 RSI, KDJ, MACD 等) ...
        'rsi_period': 12, 'rsi_oversold': 30, 'rsi_overbought': 70, 'rsi_extreme_oversold': 20, 'rsi_extreme_overbought': 80,
        'kdj_period_k': 9, 'kdj_period_d': 3, 'kdj_period_j': 3, 'kdj_oversold': 20, 'kdj_overbought': 80,
        'boll_period': 20, 'boll_std_dev': 2,
        'macd_fast': 10, 'macd_slow': 26, 'macd_signal': 9,
        'cci_period': 14, 'cci_threshold': 100, 'cci_extreme_threshold': 200,
        'mfi_period': 14, 'mfi_oversold': 20, 'mfi_overbought': 80, 'mfi_extreme_oversold': 10, 'mfi_extreme_overbought': 90,
        'roc_period': 12,
        'dmi_period': 14, 'adx_threshold': 20, 'adx_strong_threshold': 30,
        'sar_step': 0.02, 'sar_max': 0.2,
        'amount_ma_period': 20, 'obv_ma_period': 10, 'cmf_period': 20,
        'ema_period': 13, # 假设 prepare_strategy_dataframe 需要这个来决定获取哪个 EMA 列 (虽然模型里有很多列)
                          # 或者 prepare_strategy_dataframe 内部逻辑是获取所有 FIB EMA 列
        'weights': {'5': 0.1, '15': 0.4, '30': 0.3, '60': 0.2}, # 策略评分依赖的周期
        'volume_tf': '15', # 量能确认和 VWAP 获取的时间周期 (根据 prepare_strategy_dataframe 逻辑)
        'volume_confirmation': True, 'volume_confirm_boost': 1.1, 'volume_fail_penalty': 0.8, 'divergence_penalty': 0.3,
        'check_bearish_divergence': True, 'divergence_price_period': 5,
        'divergence_threshold_cmf': -0.05, 'divergence_threshold_mfi': 40,
    }
    strategy_instance = MacdRsiKdjBollEnhancedStrategy(params=strategy_params)

    # 3. 确定策略和分析所需的所有时间周期
    strategy_timeframes = strategy_instance.timeframes # 策略评分需要的周期
    all_required_timeframes = set(strategy_timeframes)
    all_required_timeframes.add(time_level_for_analysis) # 添加分析周期
    volume_tf = strategy_params['volume_tf'] # 获取量能/VWAP周期
    all_required_timeframes.add(volume_tf) # 确保量能/VWAP周期也被请求
    timeframes_list = sorted(list(all_required_timeframes), key=int) # 排序

    stock = await stock_basic_dao.get_stock_by_code(stock_code)
    if not stock:
        logger.error(f"无法找到股票信息: {stock_code}")
        return

    # 4. 准备统一的策略数据帧 (调用 IndicatorService)
    logger.info(f"[{stock}] 正在准备统一策略数据 (周期: {timeframes_list})...")
    limit_count = 1500 # 获取足够的数据量
    # 这个函数现在是核心的数据获取和整合步骤
    strategy_df: Optional[pd.DataFrame] = await indicator_service.prepare_strategy_dataframe(
        stock_code=stock_code,
        timeframes=timeframes_list, # 传递所有需要的周期
        strategy_params=strategy_params, # 传递策略参数
        limit_per_tf=limit_count
    )

    # 5. 检查数据准备是否成功
    if strategy_df is None or strategy_df.empty:
        logger.error(f"[{stock}] 统一策略数据准备失败或为空，无法继续。")
        return
    logger.info(f"[{stock}] 统一策略数据准备完成，形状: {strategy_df.shape}")
    # logger.debug(f"[{stock}] strategy_df columns: {strategy_df.columns.tolist()}") # 打印列名以供调试

    # 6. 生成评分
    logger.info(f"[{stock}] 正在生成策略评分 (0-100)...")
    scores: Optional[pd.Series] = None
    intermediate_data: Optional[pd.DataFrame] = None
    try:
        # 策略的 run 方法会使用 strategy_df 中它需要的列
        scores = strategy_instance.run(strategy_df)
        intermediate_data = strategy_instance.get_intermediate_data()
        logger.info(f"[{stock}] 策略评分生成完成。")

        # 7. 查看评分结果
        if scores is not None and not scores.empty:
            scores_display = scores.copy()
            # ... (时区转换代码，同前) ...
            if local_tz and isinstance(scores_display.index, pd.DatetimeIndex):
                try:
                    target_display_tz = ZoneInfo(str(local_tz)) if ZoneInfo else None
                    if target_display_tz:
                        # 假设 run 返回的索引与 strategy_df 一致，且 prepare_strategy_dataframe 已处理好时区 (Asia/Shanghai)
                        source_tz = scores_display.index.tz
                        if source_tz is None: # 如果是幼稚类型，假定为 Asia/Shanghai
                           source_tz = ZoneInfo("Asia/Shanghai") if ZoneInfo else None
                           if source_tz: scores_display.index = scores_display.index.tz_localize(source_tz)

                        if source_tz and source_tz != target_display_tz:
                            logger.debug(f"[{stock}] 评分索引时区为 {source_tz}，转换为 {local_tz_name}...")
                            scores_display.index = scores_display.index.tz_convert(target_display_tz)
                        elif not source_tz:
                             logger.warning(f"[{stock}] 无法确定评分源时区，不进行显示时区转换。")

                except Exception as convert_e:
                    logger.warning(f"[{stock}] 转换评分索引时区时出错: {convert_e}")

            print(f"\n[{stock}] 最新的评分 (最后10条，时间：{local_tz_name}):")
            print(scores_display.tail(10).round(2))
            print("\n评分统计描述:")
            print(scores.describe().round(2))
            nan_count = scores.isna().sum()
            if nan_count > 0: print(f"\n警告: 生成的评分中包含 {nan_count} 个 NaN 值。")

            # --- 开始进行趋势和 T+0 分析 ---
            logger.info(f"[{stock}] 开始准备分析输入 (使用已获取的数据)...")

            # a. 确定分析所需的列名
            price_col = f'close_{time_level_for_analysis}'
            # 根据 prepare_strategy_dataframe 的逻辑，VWAP 列名是基于 volume_tf 的
            vwap_col = f'vwap_{volume_tf}'

            # b. 检查所需列是否存在于 strategy_df
            analysis_input_cols = {'score': scores} # 从评分结果获取 score
            cols_ok = True
            if price_col in strategy_df.columns:
                analysis_input_cols['close'] = strategy_df[price_col]
            else:
                logger.warning(f"[{stock}] 策略 DataFrame 中缺少价格列 '{price_col}'，分析可能不完整。")
                analysis_input_cols['close'] = pd.Series(dtype=float) # 添加空列以满足 analyze_score_trend 签名
                cols_ok = False # 标记缺少核心列

            if vwap_col in strategy_df.columns:
                analysis_input_cols['vwap'] = strategy_df[vwap_col]
            else:
                logger.warning(f"[{stock}] 策略 DataFrame 中缺少 VWAP 列 '{vwap_col}'，将禁用 T+0 分析。")
                analysis_input_cols['vwap'] = pd.Series(dtype=float) # 添加空列

            # c. 创建用于分析的 DataFrame
            # 使用 pd.concat 更容易处理索引对齐问题（假设 scores 的索引与 strategy_df 一致）
            try:
                # 确保所有 Series 都有相同的索引类型和时区（如果有时区）
                common_index = scores.index
                for key, series in analysis_input_cols.items():
                    if key != 'score': # score 已经有了
                        if isinstance(series, pd.Series):
                             # 重新索引以匹配 score 的索引，填充缺失值为 NaN
                             analysis_input_cols[key] = series.reindex(common_index)
                        else: # 如果是空 Series
                             analysis_input_cols[key] = pd.Series(np.nan, index=common_index)


                score_price_vwap_df = pd.DataFrame(analysis_input_cols)
                # 清理掉 score 或 close 为 NaN 的行，因为它们对分析至关重要
                score_price_vwap_df.dropna(subset=['score', 'close'], how='any', inplace=True)

            except Exception as concat_err:
                 logger.error(f"[{stock}] 创建分析 DataFrame 时出错: {concat_err}", exc_info=True)
                 score_price_vwap_df = pd.DataFrame() # 创建空 DF 以跳过分析

            if not score_price_vwap_df.empty:
                logger.info(f"[{stock}] 分析输入数据准备完成，开始调用 analyze_score_trend...")
                # d. 定义 T+0 参数
                t0_settings = {
                    'enabled': True, # 默认启用，analyze_score_trend 内部会检查 vwap 列
                    'buy_dev_threshold': -0.003,
                    'sell_dev_threshold': 0.005
                }
                # e. 调用分析函数
                analysis_result_df = analyze_score_trend(
                    stock_code=str(stock),
                    score_price_vwap_df=score_price_vwap_df,
                    t0_params=t0_settings
                )
                # analysis_result_df 可以在这里进一步使用或保存
            else:
                logger.warning(f"[{stock}] 准备用于分析的数据为空或失败，跳过趋势分析。")

            # --- 结束趋势和 T+0 分析 ---

        else:
            logger.error(f"\n[{stock}] 未能获取有效的评分结果 (scores is None or empty)。")

        # 查看中间数据 (可选)
        if intermediate_data is not None:
             # logger.debug(f"中间计算数据 (最后5行):\n{intermediate_data.tail()}")
             pass # 可以取消注释以查看

    except ValueError as ve:
        logger.error(f"[{stock}] 生成评分或分析时发生值错误: {ve}", exc_info=True)
    except Exception as e:
        logger.error(f"[{stock}] 生成评分或分析时发生未知错误: {e}", exc_info=True)


# --- Django Management Command 类 ---
class Command(BaseCommand):
    help = '测试策略评分、趋势分析及基于 VWAP 的 T+0 信号 (假设 prepare_strategy_dataframe 已包含所有数据)'

    def add_arguments(self, parser):
        parser.add_argument('stock_code', type=str, help='要测试的股票代码 (例如: 000001)')
        parser.add_argument(
            '--level',
            type=str,
            default='5', # 默认使用 5 分钟级别进行分析
            help='用于 T+0 分析和选取 close/vwap 列的时间级别 (例如: 1, 5, 15)'
        )

    def handle(self, *args, **options):
        stock_code_to_test = options['stock_code']
        time_level = options['level']

        self.stdout.write(self.style.SUCCESS(f'开始测试策略评分及趋势分析 for {stock_code_to_test} (分析级别: {time_level})...'))

        try:
            # 运行更新后的异步测试函数
            asyncio.run(test_strategy_scores(
                stock_code=stock_code_to_test,
                time_level_for_analysis=time_level
            ))
            self.stdout.write(self.style.SUCCESS(f'测试完成 for {stock_code_to_test}.'))
        except Exception as e:
            logger.error(f"命令执行期间发生错误 for {stock_code_to_test}: {e}", exc_info=True)
            self.stderr.write(self.style.ERROR(f'测试过程中发生错误: {e}'))

