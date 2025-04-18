# tasks/management/commands/test_strategy_signals.py

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from django.core.management.base import BaseCommand
import logging
from django.utils import timezone

from dao_manager.daos.stock_realtime_dao import StockRealtimeDAO
from stock_models.stock_realtime import StockRealtimeData # 用于处理时区

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

# --- analyze_score_trend 函数 (更新版：加入 233 EMA 和反转判断) ---
async def analyze_score_trend(stock_code: str,
    score_price_vwap_df: pd.DataFrame, # 输入：包含 'score', 'close', 'vwap' 列
    t0_params: Optional[Dict[str, Any]] = None) -> Optional[pd.DataFrame]:
    """
    细化分析策略评分和价格的趋势，使用 VWAP 提供日内 T+0 交易信号提示，
    并加入 233 周期 EMA 分析和趋势反转状态判断。

    Args:
        stock_code (str): 股票代码。
        score_price_vwap_df (pd.DataFrame): 包含 'score', 'close' 列，可选 'vwap' 列的 DataFrame。
                                            索引必须是 DatetimeIndex。
        t0_params (Optional[Dict[str, Any]]): T+0 交易信号参数。

    Returns:
        Optional[pd.DataFrame]: 包含分析结果的 DataFrame，如果分析失败则返回 None。
                                新增列: 'ema_score_233', 'long_term_context', 'reversal_signal'。
    """
    # --- 参数处理 ---
    default_t0_params = {
        'enabled': True,
        'buy_dev_threshold': -0.003, # 价格低于 VWAP 0.3%
        'sell_dev_threshold': 0.005, # 价格高于 VWAP 0.5%
        'use_long_term_filter': True # 是否使用 233 EMA 作为 T+0 过滤
    }
    if t0_params is None: t0_params = default_t0_params
    else: t0_params = {**default_t0_params, **t0_params}

    # --- 输入数据验证 ---
    if ta is None: logger.warning(f"[{stock_code}] pandas_ta 未安装，无法计算 EMA。"); return None
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

    # --- 更新数据量要求 ---
    LONG_TERM_EMA_PERIOD = 233
    # 需要足够数据计算最长 EMA，并留有一定余量让 EMA 稳定
    min_required_data = LONG_TERM_EMA_PERIOD + 55 # 至少需要 233 + 55 = 288 条数据
    if len(score_price_vwap_df) < min_required_data:
        logger.warning(f"[{stock_code}] 数据点 ({len(score_price_vwap_df)}) 不足 {min_required_data}，无法进行 233 EMA 分析。")
        # 可以选择返回 None，或者继续进行不包含 233 EMA 的分析
        # 这里选择继续，但会在后续计算中产生 NaN
        # return None # 如果严格要求必须有 233 EMA 分析，则取消此行注释

    logger.info(f"[{stock_code}] 开始分析 (包含 233 EMA 和反转检测, T+0 基于 VWAP)...")
    analysis_df = score_price_vwap_df.copy()
    # --- 实例化实时数据 DAO ---
    realtime_dao = StockRealtimeDAO()

    # 1. 标准化时区 (代码同前)
    try:
        if ZoneInfo:
            target_tz = ZoneInfo("Asia/Shanghai")
            if analysis_df.index.tz is None:
                try: analysis_df.index = analysis_df.index.tz_localize(target_tz)
                except Exception: analysis_df.index = analysis_df.index.tz_localize('UTC').tz_convert(target_tz)
            elif analysis_df.index.tz != target_tz: analysis_df.index = analysis_df.index.tz_convert(target_tz)
        else: logger.warning(f"[{stock_code}] zoneinfo 不可用...")
    except Exception as e: logger.error(f"[{stock_code}] 处理时区出错: {e}", exc_info=True); return None

    # 2. 计算评分 EMA (包括 233)
    fib_periods = [5, 13, 21, 55]
    all_ema_periods = fib_periods + [LONG_TERM_EMA_PERIOD]
    try:
        for period in all_ema_periods:
            analysis_df[f'ema_score_{period}'] = ta.ema(analysis_df['score'], length=period)
    except Exception as e:
        logger.error(f"[{stock_code}] 计算评分 EMA 出错: {e}", exc_info=True)
        # 即使 EMA 计算出错，也尝试返回部分结果
        # return analysis_df # 或者根据需要决定是否返回

    # 3. 计算评分趋势排列信号 (基于 Fib EMAs)
    signal_5_13 = np.where(analysis_df['ema_score_5'] > analysis_df['ema_score_13'], 1, np.where(analysis_df['ema_score_5'] < analysis_df['ema_score_13'], -1, 0))
    signal_13_21 = np.where(analysis_df['ema_score_13'] > analysis_df['ema_score_21'], 1, np.where(analysis_df['ema_score_13'] < analysis_df['ema_score_21'], -1, 0))
    signal_21_55 = np.where(analysis_df['ema_score_21'] > analysis_df['ema_score_55'], 1, np.where(analysis_df['ema_score_21'] < analysis_df['ema_score_55'], -1, 0))
    analysis_df['alignment_signal'] = signal_5_13 + signal_13_21 + signal_21_55
    # 确保所有用于计算排列的 EMA 都存在时才计算信号
    ema_cols_for_alignment = [f'ema_score_{p}' for p in fib_periods]
    analysis_df.loc[analysis_df[ema_cols_for_alignment].isna().any(axis=1), 'alignment_signal'] = np.nan

    # 4. 计算评分趋势强度 (13 vs 55)
    analysis_df['ema_strength_13_55'] = analysis_df['ema_score_13'] - analysis_df['ema_score_55']

    # 5. 计算评分动能
    analysis_df['score_momentum'] = analysis_df['score'].diff()

    # 6. 计算评分波动性
    volatility_window = 10
    analysis_df['score_volatility'] = analysis_df['score'].rolling(window=volatility_window).std()

    # --- 新增：7. 长期趋势背景分析 (基于 233 EMA) ---
    analysis_df['long_term_context'] = np.nan # 初始化为 NaN
    if f'ema_score_{LONG_TERM_EMA_PERIOD}' in analysis_df.columns:
        # 使用当前 score 与 233 EMA 的关系判断
        analysis_df['long_term_context'] = np.where(
            analysis_df['score'] > analysis_df[f'ema_score_{LONG_TERM_EMA_PERIOD}'], 1, # 评分在长期均线之上
            np.where(analysis_df['score'] < analysis_df[f'ema_score_{LONG_TERM_EMA_PERIOD}'], -1, 0) # 评分在长期均线之下或持平
        )
        # 如果 233 EMA 本身是 NaN，则 context 也是 NaN
        analysis_df.loc[analysis_df[f'ema_score_{LONG_TERM_EMA_PERIOD}'].isna(), 'long_term_context'] = np.nan
    else:
        logger.warning(f"[{stock_code}] 未能计算 'ema_score_{LONG_TERM_EMA_PERIOD}'，无法判断长期趋势背景。")


    # --- 新增：8. 趋势反转信号检测 ---
    analysis_df['reversal_signal'] = 0 # 0: 无信号, 1: 潜在底背离/反转, -1: 潜在顶背离/反转
    # 需要至少两条 alignment_signal 数据来判断变化
    if len(analysis_df) > 1 and 'alignment_signal' in analysis_df.columns:
        prev_alignment = analysis_df['alignment_signal'].shift(1)
        current_alignment = analysis_df['alignment_signal']
        current_momentum = analysis_df['score_momentum']

        # 条件 1: 排列信号发生显著变化
        # 潜在顶部反转：之前是强多头/偏多头，现在变为中性或空头
        top_reversal_condition = ((prev_alignment >= 1) & (current_alignment <= 0))
        # 潜在底部反转：之前是强空头/偏空头，现在变为中性或多头
        bottom_reversal_condition = ((prev_alignment <= -1) & (current_alignment >= 0))

        # 条件 2: 结合动能（可选，增加信号确认度）
        # 在顶部反转迹象时，动能转负或持续为负
        top_reversal_condition = top_reversal_condition & (current_momentum <= 0)
        # 在底部反转迹象时，动能转正或持续为正
        bottom_reversal_condition = bottom_reversal_condition & (current_momentum >= 0)

        analysis_df.loc[top_reversal_condition, 'reversal_signal'] = -1
        analysis_df.loc[bottom_reversal_condition, 'reversal_signal'] = 1

        # 清理 NaN 导致的反转信号 (如果 prev_alignment 或 current_alignment 是 NaN)
        analysis_df.loc[prev_alignment.isna() | current_alignment.isna(), 'reversal_signal'] = 0


    # --- 更新：9. T+0 相关计算 (基于 VWAP，可选加入长期趋势过滤) ---
    analysis_df['t0_signal'] = 0
    if t0_params['enabled']:
        logger.debug(f"[{stock_code}] 计算 T+0 指标 (基于 VWAP)...")
        buy_dev_threshold = t0_params['buy_dev_threshold']
        sell_dev_threshold = t0_params['sell_dev_threshold']
        use_long_term_filter = t0_params['use_long_term_filter']

        if 'vwap' in analysis_df.columns:
            analysis_df['price_vwap_deviation'] = np.where(
                analysis_df['vwap'].isna() | (analysis_df['vwap'] == 0), np.nan,
                (analysis_df['close'] - analysis_df['vwap']) / analysis_df['vwap']
            )
        else:
            logger.warning(f"[{stock_code}] 'vwap' 列不存在，禁用 T+0。")
            t0_params['enabled'] = False

        if t0_params['enabled'] and 'price_vwap_deviation' in analysis_df.columns:
            is_score_uptrend = analysis_df['alignment_signal'] >= 1
            is_score_downtrend = analysis_df['alignment_signal'] <= -1
            is_price_below_vwap = analysis_df['price_vwap_deviation'] < buy_dev_threshold
            is_price_above_vwap = analysis_df['price_vwap_deviation'] > sell_dev_threshold

            # --- 加入长期趋势过滤 ---
            long_term_buy_ok = True
            long_term_sell_ok = True
            if use_long_term_filter and 'long_term_context' in analysis_df.columns:
                # 只有当长期背景为正 (评分在233之上) 时，才允许买入信号
                long_term_buy_ok = (analysis_df['long_term_context'] >= 0) # >= 0 允许中性或向上
                # 只有当长期背景为负 (评分在233之下) 时，才允许卖出信号
                long_term_sell_ok = (analysis_df['long_term_context'] <= 0) # <= 0 允许中性或向下
                # 处理 NaN 情况，如果 long_term_context 是 NaN，则不过滤
                long_term_buy_ok = long_term_buy_ok | analysis_df['long_term_context'].isna()
                long_term_sell_ok = long_term_sell_ok | analysis_df['long_term_context'].isna()
            elif use_long_term_filter:
                 logger.warning(f"[{stock_code}] T+0 配置了长期趋势过滤，但 'long_term_context' 列不存在，过滤未生效。")

            # --- 应用最终条件 ---
            buy_condition = is_score_uptrend & is_price_below_vwap & long_term_buy_ok
            sell_condition = is_score_downtrend & is_price_above_vwap & long_term_sell_ok

            analysis_df.loc[buy_condition, 't0_signal'] = 1
            analysis_df.loc[sell_condition, 't0_signal'] = -1
            logger.debug(f"[{stock_code}] T+0 信号 (基于 VWAP, 长期过滤: {use_long_term_filter}) 计算完成。")

        elif t0_params['enabled']:
            logger.warning(f"[{stock_code}] 未计算 'price_vwap_deviation'...");
            t0_params['enabled'] = False

    # 10. 综合分析与输出
    if not analysis_df.empty:
        latest_data = analysis_df.iloc[-1]
        latest_hist_time = analysis_df.index[-1]
        # --- 获取实时数据 --- (代码同前)
        latest_realtime: Optional[StockRealtimeData] = None
        latest_price: Optional[float] = None
        realtime_fetch_error = False
        if t0_params['enabled']: # 仅在启用 T+0 时获取
            try:
                logger.debug(f"[{stock_code}] 正在获取最新实时数据...")
                latest_realtime = await realtime_dao.get_latest_realtime_data(stock_code)
                if latest_realtime and latest_realtime.current_price is not None:
                    latest_price = float(latest_realtime.current_price)
                    logger.debug(f"[{stock_code}] 获取到实时价格: {latest_price} at {latest_realtime.trade_time}")
                else:
                    logger.warning(f"[{stock_code}] 未能获取到有效的最新实时价格。")
            except Exception as e:
                logger.error(f"[{stock_code}] 获取实时数据时出错: {e}", exc_info=True)
                realtime_fetch_error = True
        # ------------------------
        # --- 生成中文分析摘要 ---
        summary = f"[{stock_code}] 最新评分与价格趋势分析 (历史数据截至: {latest_hist_time.strftime('%Y-%m-%d %H:%M:%S %Z')}):\n"
        summary += f"  - 最新评分: {latest_data['score']:.2f}, 最新价格: {latest_data['close']:.2f}, 最新VWAP: {latest_data.get('vwap', 'N/A'):.2f}\n"
        summary += f"  - 评分 EMA: 5={latest_data['ema_score_5']:.2f}, 13={latest_data['ema_score_13']:.2f}, 21={latest_data['ema_score_21']:.2f}, 55={latest_data['ema_score_55']:.2f}\n"
        # --- 新增：233 EMA 信息 ---
        ema_233_val = latest_data.get(f'ema_score_{LONG_TERM_EMA_PERIOD}', np.nan)
        if pd.isna(ema_233_val):
            summary += f"  - 评分 EMA 233: 数据不足 (NaN)\n"
        else:
            summary += f"  - 评分 EMA 233: {ema_233_val:.2f}\n"
        # --- 新增：长期趋势背景解读 ---
        long_term_ctx = latest_data.get('long_term_context', np.nan)
        if pd.isna(long_term_ctx):
            summary += "  - 长期趋势背景 (基于评分 vs 233 EMA): 未知 (NaN)\n"
        elif long_term_ctx == 1:
            summary += "  - 长期趋势背景 (基于评分 vs 233 EMA): 偏多 (评分 > 233 EMA)\n"
        elif long_term_ctx == -1:
            summary += "  - 长期趋势背景 (基于评分 vs 233 EMA): 偏空 (评分 < 233 EMA)\n"
        else: # long_term_ctx == 0
            summary += "  - 长期趋势背景 (基于评分 vs 233 EMA): 中性 (评分 ≈ 233 EMA)\n"
        # --------------------------
        alignment = latest_data['alignment_signal']
        if pd.isna(alignment): summary += "  - 短期趋势排列 (5/13/21/55 EMA): 信号不足 (NaN)\n"
        elif alignment == 3: summary += "  - 短期趋势排列 (5/13/21/55 EMA): 完全多头 (+3)\n"
        elif alignment == -3: summary += "  - 短期趋势排列 (5/13/21/55 EMA): 完全空头 (-3)\n"
        elif alignment > 0: summary += f"  - 短期趋势排列 (5/13/21/55 EMA): 偏多头 ({int(alignment)})\n"
        elif alignment < 0: summary += f"  - 短期趋势排列 (5/13/21/55 EMA): 偏空头 ({int(alignment)})\n"
        else: summary += "  - 短期趋势排列 (5/13/21/55 EMA): 混合/粘合 (0)\n"
        # --- 新增：反转信号解读 ---
        reversal = latest_data.get('reversal_signal', 0)
        reversal_text = "无明显信号"
        if reversal == 1: reversal_text = "**注意：潜在底部反转信号**"
        elif reversal == -1: reversal_text = "**注意：潜在顶部反转信号**"
        summary += f"  - 趋势反转信号 (基于排列变化): {reversal_text}\n"
        # ------------------------
        strength = latest_data['ema_strength_13_55'] # 可以保留或移除，因为排列信号已包含强度信息
        # summary += f"  - 趋势强度 (EMA13 - EMA55): {strength:.2f}\n" # 可选
        momentum = latest_data['score_momentum']
        if pd.isna(momentum): summary += "  - 评分动能 (单期变化): NaN\n"
        else:
            summary += f"  - 评分动能 (单期变化): {momentum:.2f} "
            if momentum > 0.5: summary += "(显著上升)\n"
            elif momentum > 0: summary += "(上升)\n"
            elif momentum < -0.5: summary += "(显著下降)\n"
            elif momentum < 0: summary += "(下降)\n"
            else: summary += "(持平)\n"
        # --- 信号稳定性解读 (代码同前) ---
        if len(analysis_df) >= 3:
            recent_alignment = analysis_df['alignment_signal'].iloc[-3:].tolist()
            stable_signal = "未知"
            if all(a == 3 for a in recent_alignment if not pd.isna(a)): stable_signal = "稳定多头排列"
            elif all(a == -3 for a in recent_alignment if not pd.isna(a)): stable_signal = "稳定空头排列"
            elif not pd.isna(alignment):
                 if alignment > 0 and all(a > 0 for a in recent_alignment if not pd.isna(a)): stable_signal = "保持偏多头"
                 elif alignment < 0 and all(a < 0 for a in recent_alignment if not pd.isna(a)): stable_signal = "保持偏空头"
                 else: stable_signal = "信号波动"
            else: stable_signal = "信号不足"
            summary += f"  - 信号稳定性 (近3期): {stable_signal}\n"
        else: summary += "  - 信号稳定性 (近3期): 数据不足\n"
        # --- 评分波动性解读 (代码同前) ---
        volatility = latest_data['score_volatility']
        if pd.isna(volatility): summary += f"  - 评分波动性 ({volatility_window}期 std): NaN\n"
        else:
            summary += f"  - 评分波动性 ({volatility_window}期 std): {volatility:.2f} "
            try:
                if len(analysis_df['score_volatility'].dropna()) > volatility_window * 2:
                    q75 = analysis_df['score_volatility'].quantile(0.75)
                    q25 = analysis_df['score_volatility'].quantile(0.25)
                    if volatility > q75: summary += "(偏高)\n"
                    elif volatility < q25: summary += "(偏低)\n"
                    else: summary += "(适中)\n"
                else: summary += "(历史数据不足无法判断高低)\n"
            except Exception: summary += "(无法计算历史分位数)\n"
        # --- 更新：T+0 信号摘要 (基于实时价格和可选的长期过滤) ---
        summary += f"--- 日内 T+0 交易信号 (基于实时价格 vs 最新历史 VWAP, 长期过滤: {'启用' if t0_params.get('use_long_term_filter', False) else '禁用'}) ---\n"
        if t0_params['enabled']:
            latest_vwap = latest_data.get('vwap', np.nan)
            if realtime_fetch_error: summary += "  - 实时状态: 获取实时数据失败\n  - T+0 信号: 无法判断\n"
            elif latest_price is None: summary += "  - 实时状态: 未获取到有效实时价格\n  - T+0 信号: 无法判断\n"
            elif pd.isna(latest_vwap) or latest_vwap == 0:
                 summary += f"  - 实时价格: {latest_price:.2f}\n"
                 summary += "  - 最新历史 VWAP: 无效或为零\n"
                 summary += "  - T+0 信号: 无法判断 (VWAP无效)\n"
            else:
                current_deviation = (latest_price - latest_vwap) / latest_vwap
                summary += f"  - 实时价格: {latest_price:.2f} (时间: {latest_realtime.trade_time if latest_realtime else 'N/A'})\n"
                summary += f"  - 最新历史 VWAP: {latest_vwap:.2f}\n"
                summary += f"  - 当前价格相对 VWAP 偏离度: {current_deviation:.2%}\n"

                # 判断 T+0 信号 (使用最新的历史排列和长期背景)
                latest_alignment = latest_data['alignment_signal']
                latest_long_term_ctx = latest_data.get('long_term_context', np.nan) # 获取最新的长期背景

                if pd.isna(latest_alignment):
                     summary += "  - T+0 信号: 无法判断 (评分趋势信号不足)\n"
                else:
                    buy_threshold = t0_params['buy_dev_threshold']
                    sell_threshold = t0_params['sell_dev_threshold']
                    use_filter = t0_params.get('use_long_term_filter', False)

                    # 检查买入条件
                    potential_buy = latest_alignment >= 1 and current_deviation < buy_threshold
                    buy_filter_passed = True
                    if use_filter:
                        if pd.isna(latest_long_term_ctx):
                            buy_filter_passed = False # 长期趋势未知时不允许操作 (或者可以允许，取决于策略)
                            summary += "      (买入过滤: 长期趋势未知, 信号阻止)\n"
                        elif latest_long_term_ctx < 0: # 长期趋势向下时不允许买入
                            buy_filter_passed = False
                            summary += "      (买入过滤: 长期趋势偏空, 信号阻止)\n"

                    # 检查卖出条件
                    potential_sell = latest_alignment <= -1 and current_deviation > sell_threshold
                    sell_filter_passed = True
                    if use_filter:
                        if pd.isna(latest_long_term_ctx):
                            sell_filter_passed = False
                            summary += "      (卖出过滤: 长期趋势未知, 信号阻止)\n"
                        elif latest_long_term_ctx > 0: # 长期趋势向上时不允许卖出
                            sell_filter_passed = False
                            summary += "      (卖出过滤: 长期趋势偏多, 信号阻止)\n"

                    # 输出最终信号
                    if potential_buy and buy_filter_passed:
                        summary += f"  - T+0 信号: **潜在买入点** (短期趋势向好, 价<VWAP阈值{buy_threshold:.2%}, 长期趋势允许)\n"
                    elif potential_sell and sell_filter_passed:
                        summary += f"  - T+0 信号: **潜在卖出点** (短期趋势向差, 价>VWAP阈值{sell_threshold:.2%}, 长期趋势允许)\n"
                    else:
                        summary += "  - T+0 信号: 无或观望\n"
                        # 解释无信号原因
                        if latest_alignment >= 1 and current_deviation >= buy_threshold: summary += "      (原因: 短期趋势向好，但价格未低于买入阈值)\n"
                        elif latest_alignment <= -1 and current_deviation <= sell_threshold: summary += "      (原因: 短期趋势向差，但价格未高于卖出阈值)\n"
                        elif latest_alignment == 0: summary += "      (原因: 短期趋势不明朗)\n"
                        elif potential_buy and not buy_filter_passed and use_filter: summary += f"      (原因: 满足价格条件但被长期趋势 ({'未知' if pd.isna(latest_long_term_ctx) else ('偏空' if latest_long_term_ctx < 0 else '中性')}) 过滤)\n"
                        elif potential_sell and not sell_filter_passed and use_filter: summary += f"      (原因: 满足价格条件但被长期趋势 ({'未知' if pd.isna(latest_long_term_ctx) else ('偏多' if latest_long_term_ctx > 0 else '中性')}) 过滤)\n"
        else:
            summary += "--- 日内 T+0 交易信号: 未启用或因数据缺失无法计算 ---\n"
        # --------------------------
        # 打印摘要
        print("\n" + "="*30 + " 评分与价格趋势分析摘要 " + "="*30)
        print(summary)
        print("="*78)

        # (可选) 打印 DataFrame 尾部数据，包含新列
        # print(f"\n[{stock_code}] 详细历史趋势分析数据 (最后10条，时间：Asia/Shanghai):")
        # display_cols = ['score', 'close']
        # if 'vwap' in analysis_df.columns: display_cols.append('vwap')
        # display_cols.extend([f'ema_score_{p}' for p in all_ema_periods if f'ema_score_{p}' in analysis_df.columns])
        # display_cols.extend(['alignment_signal', 'long_term_context', 'reversal_signal', 'ema_strength_13_55', 'score_momentum', 'score_volatility'])
        # if t0_params['enabled'] and 'price_vwap_deviation' in analysis_df.columns:
        #     display_cols.append('price_vwap_deviation')
        # display_cols.append('t0_signal') # 显示历史 T+0 信号
        # display_cols = [col for col in display_cols if col in analysis_df.columns]
        # if display_cols:
        #     with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', 1500, 'display.float_format', '{:.2f}'.format):
        #          print(analysis_df[display_cols].tail(10).to_string())
        # else:
        #     logger.warning(f"[{stock_code}] 无法显示详细历史趋势分析数据...")

    else: logger.warning(f"[{stock_code}] 分析结果 DF 为空..."); return None

    # 11. (可选) 保存结果到 Redis (确保新列可序列化)
    try:
        cache_key = f"strategy_score_trend:{stock_code}"
        # 重置索引并将 datetime 转换为 ISO 格式字符串
        analysis_df_json = analysis_df.reset_index()
        # 检查索引列名，通常是 'trade_time' 或 'index'
        index_col_name = analysis_df_json.columns[0] # 假设第一列是原来的索引
        if pd.api.types.is_datetime64_any_dtype(analysis_df_json[index_col_name]):
             analysis_df_json[index_col_name] = analysis_df_json[index_col_name].dt.strftime('%Y-%m-%dT%H:%M:%S%z')

        # 将 NaN 替换为 None 以便 JSON 序列化
        analysis_df_serializable = analysis_df_json.replace({np.nan: None, pd.NaT: None})
        # 转换整数列中的 None 为特定值或确保它们是浮点数（如果需要）
        # 例如，reversal_signal 和 t0_signal 可能是整数，但 NaN 会变 None
        for col in ['alignment_signal', 'long_term_context', 'reversal_signal', 't0_signal']:
             if col in analysis_df_serializable.columns:
                 # 转换为浮点数可以保留 None/NaN
                 analysis_df_serializable[col] = analysis_df_serializable[col].astype(float)


        cache.set(cache_key, analysis_df_serializable.to_json(orient='records', date_format='iso'), timeout=60 * 60)
        logger.info(f"[{stock_code}] 分析结果已缓存至 Redis (Key: {cache_key})。")
    except Exception as e: logger.error(f"[{stock_code}] 缓存分析结果至 Redis 出错: {e}", exc_info=True)

    return analysis_df
# --- 结束 analyze_score_trend 函数 ---

# --- test_strategy_scores 函数 (无需修改，因为它调用更新后的 analyze_score_trend) ---
async def test_strategy_scores(stock_code: str, time_level_for_analysis: str = '5'):
    """
    测试指定股票代码的策略评分生成过程，并进行趋势和 T+0 分析。
    假设 prepare_strategy_dataframe 已返回包含所有所需指标 (含 EMA, VWAP) 的 DataFrame。

    Args:
        stock_code (str): 股票代码。
        time_level_for_analysis (str): 用于 T+0 分析和选取 close/vwap 列的时间周期 (例如 '1', '5', '15')。
                                       策略评分本身可能依赖其他周期。
    """
    # --- 获取本地时区 --- (代码同前)
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

    # 1. 初始化服务和 DAO 实例 (代码同前)
    indicator_service = IndicatorService()
    stock_basic_dao = StockBasicDAO()

    # 2. 定义策略参数 (代码同前)
    strategy_params: Dict[str, Any] = {
        # ... (保持原有参数) ...
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
        'ema_period': 13,
        'weights': {'5': 0.1, '15': 0.4, '30': 0.3, '60': 0.2},
        'volume_tf': '15',
        'volume_confirmation': True, 'volume_confirm_boost': 1.1, 'volume_fail_penalty': 0.8, 'divergence_penalty': 0.3,
        'check_bearish_divergence': True, 'divergence_price_period': 5,
        'divergence_threshold_cmf': -0.05, 'divergence_threshold_mfi': 40,
    }
    strategy_instance = MacdRsiKdjBollEnhancedStrategy(params=strategy_params)

    # 3. 确定策略和分析所需的所有时间周期 (代码同前)
    strategy_timeframes = strategy_instance.timeframes
    all_required_timeframes = set(strategy_timeframes)
    all_required_timeframes.add(time_level_for_analysis)
    volume_tf = strategy_params['volume_tf']
    all_required_timeframes.add(volume_tf)
    timeframes_list = sorted(list(all_required_timeframes), key=int)

    stock = await stock_basic_dao.get_stock_by_code(stock_code)
    if not stock:
        logger.error(f"无法找到股票信息: {stock_code}")
        return

    # 4. 准备统一的策略数据帧 (调用 IndicatorService)
    logger.info(f"[{stock}] 正在准备统一策略数据 (周期: {timeframes_list})...")
    # --- 增加数据量以满足 233 EMA 计算 ---
    # 需要确保 prepare_strategy_dataframe 能够获取足够的数据
    # 假设 limit_per_tf 控制每个周期的条数，我们需要保证基础周期（如1分钟）有足够数据
    # 如果 prepare_strategy_dataframe 内部做了聚合，需要保证聚合前的数据量足够
    # 这里设置一个较大的 limit_count，具体效果取决于 prepare_strategy_dataframe 实现
    limit_count = 2000 # 增加请求的数据量
    strategy_df: Optional[pd.DataFrame] = await indicator_service.prepare_strategy_dataframe(
        stock_code=stock_code,
        timeframes=timeframes_list,
        strategy_params=strategy_params,
        limit_per_tf=limit_count
    )

    # 5. 检查数据准备是否成功 (代码同前)
    if strategy_df is None or strategy_df.empty:
        logger.error(f"[{stock}] 统一策略数据准备失败或为空，无法继续。")
        return
    logger.info(f"[{stock}] 统一策略数据准备完成，形状: {strategy_df.shape}")
    # logger.debug(f"[{stock}] strategy_df columns: {strategy_df.columns.tolist()}")

    # 6. 生成评分 (代码同前)
    logger.info(f"[{stock}] 正在生成策略评分 (0-100)...")
    scores: Optional[pd.Series] = None
    intermediate_data: Optional[pd.DataFrame] = None
    try:
        scores = strategy_instance.run(strategy_df)
        intermediate_data = strategy_instance.get_intermediate_data()
        logger.info(f"[{stock}] 策略评分生成完成。")

        # 7. 查看评分结果和进行分析 (代码同前，但会调用更新后的 analyze_score_trend)
        if scores is not None and not scores.empty:
            scores_display = scores.copy()
            # ... (时区转换代码) ...
            if local_tz and isinstance(scores_display.index, pd.DatetimeIndex):
                 # ... (代码同前) ...
                 pass # 省略重复代码

            print(f"\n[{stock}] 最新的评分 (最后10条，时间：{local_tz_name}):")
            print(scores_display.tail(10).round(2))
            print("\n评分统计描述:")
            print(scores.describe().round(2))
            nan_count = scores.isna().sum()
            if nan_count > 0: print(f"\n警告: 生成的评分中包含 {nan_count} 个 NaN 值。")

            # --- 开始进行趋势和 T+0 分析 ---
            logger.info(f"[{stock}] 开始准备分析输入 (使用已获取的数据)...")

            # a. 确定分析所需的列名 (代码同前)
            price_col = f'close_{time_level_for_analysis}'
            vwap_col = f'vwap_{volume_tf}'

            # b. 检查所需列是否存在于 strategy_df (代码同前)
            analysis_input_cols = {'score': scores}
            cols_ok = True
            if price_col in strategy_df.columns: analysis_input_cols['close'] = strategy_df[price_col]
            else:
                logger.warning(f"[{stock}] 策略 DataFrame 中缺少价格列 '{price_col}'，分析可能不完整。")
                analysis_input_cols['close'] = pd.Series(dtype=float)
                cols_ok = False

            if vwap_col in strategy_df.columns: analysis_input_cols['vwap'] = strategy_df[vwap_col]
            else:
                logger.warning(f"[{stock}] 策略 DataFrame 中缺少 VWAP 列 '{vwap_col}'，将禁用 T+0 分析。")
                analysis_input_cols['vwap'] = pd.Series(dtype=float)

            # c. 创建用于分析的 DataFrame (代码同前)
            try:
                common_index = scores.index
                for key, series in analysis_input_cols.items():
                    if key != 'score':
                        if isinstance(series, pd.Series):
                             analysis_input_cols[key] = series.reindex(common_index)
                        else:
                             analysis_input_cols[key] = pd.Series(np.nan, index=common_index)

                score_price_vwap_df = pd.DataFrame(analysis_input_cols)
                score_price_vwap_df.dropna(subset=['score', 'close'], how='any', inplace=True)

            except Exception as concat_err:
                 logger.error(f"[{stock}] 创建分析 DataFrame 时出错: {concat_err}", exc_info=True)
                 score_price_vwap_df = pd.DataFrame()

            if not score_price_vwap_df.empty:
                logger.info(f"[{stock}] 分析输入数据准备完成 (数据条数: {len(score_price_vwap_df)})，开始调用 analyze_score_trend...")
                # d. 定义 T+0 参数 (可以从配置文件或数据库读取)
                t0_settings = {
                    'enabled': True,
                    'buy_dev_threshold': -0.003,
                    'sell_dev_threshold': 0.005,
                    'use_long_term_filter': True # 启用长期趋势过滤
                }
                # e. 调用更新后的分析函数
                analysis_result_df = await analyze_score_trend(
                    stock_code=str(stock_code),
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
             pass

    except ValueError as ve:
        logger.error(f"[{stock}] 生成评分或分析时发生值错误: {ve}", exc_info=True)
    except Exception as e:
        logger.error(f"[{stock}] 生成评分或分析时发生未知错误: {e}", exc_info=True)


# --- Django Management Command 类 (无需修改) ---
class Command(BaseCommand):
    help = '测试策略评分、趋势分析及基于 VWAP 的 T+0 信号 (包含 233 EMA 和反转检测)'

    def add_arguments(self, parser):
        parser.add_argument('stock_code', type=str, help='要测试的股票代码 (例如: 000001)')
        parser.add_argument(
            '--level',
            type=str,
            default='5',
            help='用于 T+0 分析和选取 close/vwap 列的时间级别 (例如: 1, 5, 15)'
        )

    def handle(self, *args, **options):
        stock_code_to_test = options['stock_code']
        time_level = options['level']

        self.stdout.write(self.style.SUCCESS(f'开始测试策略评分及趋势分析 for {stock_code_to_test} (分析级别: {time_level})...'))

        try:
            asyncio.run(test_strategy_scores(
                stock_code=stock_code_to_test,
                time_level_for_analysis=time_level
            ))
            self.stdout.write(self.style.SUCCESS(f'测试完成 for {stock_code_to_test}.'))
        except Exception as e:
            logger.error(f"命令执行期间发生错误 for {stock_code_to_test}: {e}", exc_info=True)
            self.stderr.write(self.style.ERROR(f'测试过程中发生错误: {e}'))

