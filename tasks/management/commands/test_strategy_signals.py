# tasks/management/commands/test_strategy_signals.py

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from django.core.management.base import BaseCommand
import logging
from django.utils import timezone
from decimal import Decimal, InvalidOperation # 用于精确数字处理
from asgiref.sync import sync_to_async # 用于在异步代码中调用同步 ORM 操作
import tzlocal
from zoneinfo import ZoneInfo
import pandas_ta as ta

from dao_manager.daos.stock_basic_dao import StockBasicDAO
from dao_manager.daos.stock_realtime_dao import StockRealtimeDAO
from stock_models.stock_analytics import StockScoreAnalysis
from stock_models.stock_basic import StockInfo
from stock_models.stock_realtime import StockRealtimeData
from utils.cache_manager import CacheManager # 用于处理时区

# --- 导入项目模块 ---
from services.indicator_services import IndicatorService # 依赖这个服务来准备数据
from strategies.macd_rsi_kdj_boll_strategy import MacdRsiKdjBollEnhancedStrategy
# 不再需要单独导入 IndicatorDAO 来获取 VWAP
from django.core.cache import cache # 如果 CacheManager 不可用，可以考虑 Django cache

logger = logging.getLogger(__name__)

# --- 常量定义 ---
# 可以考虑将这些移到配置文件或 settings.py
DEFAULT_T0_PARAMS = {
    'enabled': True,
    'buy_dev_threshold': -0.003,
    'sell_dev_threshold': 0.005,
    'use_long_term_filter': True
}
EMA_PERIODS = [5, 13, 21, 55, 233]
LONG_TERM_EMA_PERIOD = 233
VOLATILITY_WINDOW = 10
DB_SAVE_RECENT_N = 10 # 保存最近 N 条有效记录
REDIS_CACHE_TIMEOUT = 3600 # 1 小时

# --- 辅助函数：清理潜在的无效数值 (稍作简化) ---
def clean_value(value, default=None):
    """
    将 NaN, inf, -inf, NaT, None 等无效值转换为指定的默认值 (通常是 None)。
    处理 numpy 类型和 Decimal 的特殊情况。
    """
    # 优先处理 Decimal 的特殊值
    if isinstance(value, Decimal) and (value.is_nan() or value.is_infinite()):
        return default
    # pd.isna 涵盖 None, NaN, NaT
    # np.isfinite 涵盖 float/numpy 的 inf/-inf (对非数字类型返回 False)
    if pd.isna(value) or (isinstance(value, (float, np.number)) and not np.isfinite(value)):
        return default

    # 尝试转换 numpy 数值类型为 Python 内建类型
    if isinstance(value, np.number):
        try:
            return value.item()
        except (ValueError, TypeError):
            return default # 转换失败

    # 其他情况，假设值是有效的
    return value

def format_value(value: Any, fmt_spec: str, default_str: str = "N/A") -> str:
    """
    清理值，如果结果是数字则按指定格式格式化，否则返回默认字符串。
    """
    cleaned = clean_value(value, default_str)
    # 检查清理后的值是否是数字类型 (int, float, Decimal)
    if isinstance(cleaned, (int, float, Decimal)):
        try:
            # 应用格式化字符串
            return f"{cleaned:{fmt_spec}}"
        except ValueError:
            # 如果格式化失败（虽然不太可能对于 '.2f'），回退到简单字符串转换
            return str(cleaned)
    # 如果清理后的值不是数字 (即它是 default_str)，直接返回
    return cleaned

# --- analyze_score_trend 的辅助函数 ---

def _validate_input_data(stock_code: str, df: pd.DataFrame, t0_enabled: bool) -> bool:
    """验证输入 DataFrame 的有效性。"""
    if df is None or df.empty:
        logger.warning(f"[{stock_code}] 输入的 DataFrame 为空，无法分析。")
        return False
    required_cols = ['score', 'close']
    if t0_enabled:
        required_cols.append('vwap')
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"[{stock_code}] 输入 DataFrame 缺少必需列: {missing_cols}。")
        if 'score' in missing_cols or 'close' in missing_cols:
            return False # 核心列缺失，无法继续
        # 如果缺 vwap 但 t0 启用，会在主函数中禁用 t0
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning(f"[{stock_code}] 输入 DataFrame 的索引不是 DatetimeIndex。")
        return False
    return True

def _determine_analysis_level(stock_code: str, df: pd.DataFrame, provided_level: Optional[str]) -> str:
    """确定分析的时间级别，优先使用传入的级别，否则尝试自动检测。"""
    if provided_level:
        logger.info(f"[{stock_code}] 使用外部传入的分析级别: '{provided_level}'")
        level = str(provided_level)
        return level if level.isdigit() else level.upper()

    logger.debug(f"[{stock_code}] analysis_time_level 未提供，尝试自动检测...")
    close_suffix_cols = [col for col in df.columns if col.startswith('close_')]
    for col in close_suffix_cols:
        parts = col.split('_')
        if len(parts) > 1:
            level_str = parts[-1]
            if level_str.isdigit():
                logger.info(f"[{stock_code}] 从列 '{col}' 自动检测到数字级别: '{level_str}'")
                return str(level_str)
            elif level_str.upper() in ['D', 'W', 'M']:
                logger.info(f"[{stock_code}] 从列 '{col}' 自动检测到字符级别: '{level_str.upper()}'")
                return level_str.upper()

    if 'close' in df.columns:
        logger.info(f"[{stock_code}] 未从 'close_' 列检测到级别，找到 'close' 列，假设为日线 ('D')。")
        return 'D'

    logger.warning(f"[{stock_code}] 无法自动推断分析时间级别。将使用 'unknown'。")
    return 'unknown'

def _standardize_timezone(stock_code: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """确保 DataFrame 索引使用上海时区。"""
    try:
        target_tz = ZoneInfo("Asia/Shanghai")
        if df.index.tz is None:
            # 尝试本地化，处理歧义时间
            df.index = df.index.tz_localize(target_tz, ambiguous='infer', nonexistent='shift_forward')
        elif df.index.tz != target_tz:
            df.index = df.index.tz_convert(target_tz)
        return df
    except ImportError:
         logger.warning(f"[{stock_code}] 'zoneinfo' 库不可用，无法进行精确的时区处理。")
         # 可以考虑添加 pytz 作为备选
         return df # 返回原始 DataFrame，后续操作可能受影响
    except Exception as e:
        logger.error(f"[{stock_code}] 处理 DataFrame 索引时区时出错: {e}", exc_info=True)
        return None

def _calculate_indicators(stock_code: str, df: pd.DataFrame) -> pd.DataFrame:
    """(修正版) 计算所有基于评分的指标 (EMAs, 信号, 强度等)。"""
    if ta is None:
        logger.warning(f"[{stock_code}] pandas_ta 未安装，跳过指标计算。")
        return df

    # 1. 计算 EMAs (保持不变)
    try:
        for period in EMA_PERIODS:
            df[f'ema_score_{period}'] = ta.ema(df['score'], length=period)
    except Exception as e:
        logger.error(f"[{stock_code}] 计算评分 EMA 时出错: {e}", exc_info=True)

    # 2. 计算排列信号 (保持不变)
    ema_cols_alignment = [f'ema_score_{p}' for p in [5, 13, 21, 55]]
    if all(col in df.columns for col in ema_cols_alignment):
        signal_5_13 = np.where(df['ema_score_5'] > df['ema_score_13'], 1, np.where(df['ema_score_5'] < df['ema_score_13'], -1, 0))
        signal_13_21 = np.where(df['ema_score_13'] > df['ema_score_21'], 1, np.where(df['ema_score_13'] < df['ema_score_21'], -1, 0))
        signal_21_55 = np.where(df['ema_score_21'] > df['ema_score_55'], 1, np.where(df['ema_score_21'] < df['ema_score_55'], -1, 0))
        df['alignment_signal'] = signal_5_13 + signal_13_21 + signal_21_55
        # 使用 notna() 检查 NaN 或 None
        df.loc[df[ema_cols_alignment].isna().any(axis=1), 'alignment_signal'] = np.nan
    else:
        df['alignment_signal'] = np.nan
        logger.warning(f"[{stock_code}] 缺少计算排列信号所需的 EMA 列，'alignment_signal' 将为 NaN。")

    # 3. 计算趋势强度 (保持不变)
    if 'ema_score_13' in df.columns and 'ema_score_55' in df.columns:
        df['ema_strength_13_55'] = df['ema_score_13'] - df['ema_score_55']
    else:
        df['ema_strength_13_55'] = np.nan

    # 4. 计算评分动能 (保持不变)
    df['score_momentum'] = df['score'].diff()

    # 5. 计算评分波动性 (保持不变)
    if len(df) >= VOLATILITY_WINDOW:
        df['score_volatility'] = df['score'].rolling(window=VOLATILITY_WINDOW).std()
    else:
        df['score_volatility'] = np.nan # 数据不足

    # --- 6. 长期趋势背景 (vs EMA 233) - 修正部分 ---
    ema_233_col = f'ema_score_{LONG_TERM_EMA_PERIOD}'
    df['long_term_context'] = np.nan # 初始化为 NaN

    if ema_233_col in df.columns:
        # 创建一个布尔掩码，标记 score 和 ema_233 都是有效数值（非 NaN 且非 None）的行
        # 使用 pd.to_numeric 强制转换并检查是否为 NaN，可以同时处理 None 和非数字字符串
        score_is_numeric = pd.to_numeric(df['score'], errors='coerce').notna()
        ema_233_is_numeric = pd.to_numeric(df[ema_233_col], errors='coerce').notna()
        valid_comparison_mask = score_is_numeric & ema_233_is_numeric

        # 仅在 score 和 ema_233 都是有效数字的行上执行比较和 np.select
        # .loc[valid_comparison_mask] 确保只操作有效行
        df.loc[valid_comparison_mask, 'long_term_context'] = np.select(
            [
                df.loc[valid_comparison_mask, 'score'] > df.loc[valid_comparison_mask, ema_233_col],
                df.loc[valid_comparison_mask, 'score'] < df.loc[valid_comparison_mask, ema_233_col]
            ],
            [1, -1],
            default=0 # default=0 表示 score == ema_233 (仅在 valid_comparison_mask 为 True 时应用)
        )
        # 对于 valid_comparison_mask 为 False 的行，long_term_context 保持初始化的 NaN 值
    else:
        # ema_233_col 列不存在，long_term_context 保持初始化的 NaN 值
        logger.warning(f"[{stock_code}] 未找到 '{ema_233_col}' 列，无法判断长期趋势背景。")
    # --- 修正结束 ---

    # 7. 趋势反转信号 (保持不变)
    df['reversal_signal'] = 0
    if 'alignment_signal' in df.columns and len(df) > 1:
        prev_alignment = df['alignment_signal'].shift(1)
        current_alignment = df['alignment_signal']
        # 使用 notna() 检查 NaN 或 None
        valid_mask = prev_alignment.notna() & current_alignment.notna()
        top_reversal = valid_mask & (prev_alignment >= 1) & (current_alignment <= 0)
        bottom_reversal = valid_mask & (prev_alignment <= -1) & (current_alignment >= 0)
        df.loc[top_reversal, 'reversal_signal'] = -1
        df.loc[bottom_reversal, 'reversal_signal'] = 1

    return df

def _calculate_historical_t0(stock_code: str, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """计算历史数据的 T+0 信号。"""
    df['t0_signal'] = 0
    df['price_vwap_deviation'] = np.nan

    if not params['enabled']:
        logger.info(f"[{stock_code}] T+0 功能已禁用，跳过历史 T+0 计算。")
        return df
    if 'vwap' not in df.columns:
        logger.warning(f"[{stock_code}] 'vwap' 列不存在，无法计算历史 T+0 指标。")
        return df

    logger.debug(f"[{stock_code}] 计算历史 T+0 指标...")
    # 计算价格偏离度，处理 VWAP 为 0 或 NaN 的情况
    df['price_vwap_deviation'] = np.where(
        df['vwap'].isna() | (df['vwap'] == 0),
        np.nan,
        (df['close'] - df['vwap']) / df['vwap']
    )

    # 定义 T+0 条件
    is_score_uptrend = df.get('alignment_signal', pd.Series(np.nan, index=df.index)) >= 1
    is_score_downtrend = df.get('alignment_signal', pd.Series(np.nan, index=df.index)) <= -1
    is_price_below_vwap = df['price_vwap_deviation'] < params['buy_dev_threshold']
    is_price_above_vwap = df['price_vwap_deviation'] > params['sell_dev_threshold']

    # 长期趋势过滤条件
    long_term_buy_ok = True
    long_term_sell_ok = True
    if params['use_long_term_filter']:
        if 'long_term_context' in df.columns:
            # 允许买入：长期趋势为正(1)或中性(0) 或 未知(NaN)
            long_term_buy_ok = (df['long_term_context'] >= 0) | df['long_term_context'].isna()
            # 允许卖出：长期趋势为负(-1)或中性(0) 或 未知(NaN)
            long_term_sell_ok = (df['long_term_context'] <= 0) | df['long_term_context'].isna()
        else:
            logger.warning(f"[{stock_code}] T+0 配置了长期过滤，但 'long_term_context' 列缺失，过滤未生效。")

    # 合并条件
    buy_condition = is_score_uptrend & is_price_below_vwap & long_term_buy_ok
    sell_condition = is_score_downtrend & is_price_above_vwap & long_term_sell_ok

    df.loc[buy_condition, 't0_signal'] = 1
    df.loc[sell_condition, 't0_signal'] = -1
    # 确保信号列是整数类型，处理可能的 NaN (例如 alignment_signal 为 NaN 时)
    df['t0_signal'] = df['t0_signal'].fillna(0).astype(int)

    logger.debug(f"[{stock_code}] 历史 T+0 信号计算完成 (长期过滤: {params['use_long_term_filter']})。")
    return df

async def _fetch_and_process_realtime(stock_code: str, realtime_dao: StockRealtimeDAO, latest_hist_row: pd.Series, params: Dict[str, Any]) -> Dict[str, Any]:
    """获取实时数据并计算当前 T+0 信号和相关指标。"""
    realtime_cache = {'fetch_error': True, 'fetch_time': timezone.now().isoformat()} # 默认失败
    if not params['enabled']:
        realtime_cache['fetch_error'] = False # T+0 未启用，不算错误
        realtime_cache['message'] = "T+0 feature disabled"
        return realtime_cache

    latest_vwap = clean_value(latest_hist_row.get('vwap'))
    latest_alignment = clean_value(latest_hist_row.get('alignment_signal'))
    latest_long_term_ctx = clean_value(latest_hist_row.get('long_term_context'))

    if latest_vwap is None or latest_vwap == 0:
        logger.warning(f"[{stock_code}] 最新的历史 VWAP 无效 ({latest_vwap})，无法计算实时 T+0 信号。")
        realtime_cache['fetch_error'] = False # VWAP 无效不算获取错误
        realtime_cache['message'] = "Invalid latest VWAP for comparison"
        realtime_cache['latest_vwap_used'] = latest_vwap
        return realtime_cache

    try:
        logger.debug(f"[{stock_code}] 正在获取最新实时数据...")
        latest_realtime = await realtime_dao.get_latest_realtime_data(stock_code)

        if latest_realtime and latest_realtime.current_price is not None:
            latest_price = clean_value(float(latest_realtime.current_price))
            if latest_price is None: # clean_value 可能返回 None
                 raise ValueError("Cleaned latest price is None")

            logger.debug(f"[{stock_code}] 获取到实时价格: {latest_price} at {latest_realtime.trade_time}")

            current_deviation = (latest_price - latest_vwap) / latest_vwap
            current_t0_signal = 0

            if latest_alignment is not None: # 必须有有效的短期信号
                buy_threshold = params['buy_dev_threshold']
                sell_threshold = params['sell_dev_threshold']
                use_filter = params['use_long_term_filter']

                potential_buy = latest_alignment >= 1 and current_deviation < buy_threshold
                potential_sell = latest_alignment <= -1 and current_deviation > sell_threshold

                # 长期过滤判断 (None 或 >= 0 允许买, None 或 <= 0 允许卖)
                buy_filter_passed = not use_filter or (latest_long_term_ctx is None or latest_long_term_ctx >= 0)
                sell_filter_passed = not use_filter or (latest_long_term_ctx is None or latest_long_term_ctx <= 0)

                if potential_buy and buy_filter_passed: current_t0_signal = 1
                elif potential_sell and sell_filter_passed: current_t0_signal = -1

            # 准备缓存数据
            trade_time_str = None
            if latest_realtime.trade_time:
                trade_time_str = latest_realtime.trade_time.isoformat() if hasattr(latest_realtime.trade_time, 'isoformat') else str(latest_realtime.trade_time)

            realtime_cache.update({
                'fetch_time': timezone.now().isoformat(),
                'realtime_price': latest_price,
                'realtime_trade_time': trade_time_str,
                'latest_vwap_used': latest_vwap,
                'current_deviation': current_deviation,
                'current_t0_signal': current_t0_signal,
                'fetch_error': False # 获取成功
            })
        else:
            logger.warning(f"[{stock_code}] 未能获取到有效的最新实时价格。")
            realtime_cache['fetch_error'] = False # 获取不到价格不算连接错误
            realtime_cache['message'] = "No valid real-time price received"

    except InvalidOperation:
         logger.error(f"[{stock_code}] 实时价格转换为 Decimal 时出错: {latest_realtime.current_price}", exc_info=True)
         # fetch_error 保持 True
    except Exception as e:
        logger.error(f"[{stock_code}] 获取或处理实时数据时出错: {e}", exc_info=True)
        # fetch_error 保持 True

    return realtime_cache

def _generate_summary(stock_code: str, latest_hist_row: pd.Series, realtime_cache: Dict[str, Any], analysis_df: pd.DataFrame, t0_params: Dict[str, Any], analysis_level: str) -> str:
    """(修正版) 生成分析摘要文本。"""
    summary_lines = []
    latest_hist_time = latest_hist_row.name

    summary_lines.append(f"[{stock_code}] 最新评分与价格趋势分析 (级别: {analysis_level}, 历史截至: {latest_hist_time.strftime('%Y-%m-%d %H:%M:%S %Z')})")

    # --- 使用 format_value 进行格式化 ---
    summary_lines.append(f"  - 最新历史:")
    summary_lines.append(f"    - 评分: {format_value(latest_hist_row.get('score'), '.2f')}")
    summary_lines.append(f"    - 收盘价: {format_value(latest_hist_row.get('close'), '.2f')}")
    summary_lines.append(f"    - VWAP: {format_value(latest_hist_row.get('vwap'), '.2f')}")

    summary_lines.append(f"  - 评分 EMA:")
    emas_short_mid = [f"{p}={format_value(latest_hist_row.get(f'ema_score_{p}'), '.2f')}" for p in EMA_PERIODS if p != LONG_TERM_EMA_PERIOD]
    summary_lines.append(f"    - 短中期: {', '.join(emas_short_mid)}")
    summary_lines.append(f"    - 长期 {LONG_TERM_EMA_PERIOD}: {format_value(latest_hist_row.get(f'ema_score_{LONG_TERM_EMA_PERIOD}'), '.2f')}") # <--- 使用 format_value

    # 长期趋势 (不需要格式化)
    long_term_ctx = latest_hist_row.get('long_term_context')
    ctx_text = "未知"
    if not pd.isna(long_term_ctx):
        if long_term_ctx == 1: ctx_text = "偏多 (评分 > EMA233)"
        elif long_term_ctx == -1: ctx_text = "偏空 (评分 < EMA233)"
        else: ctx_text = "中性 (评分 ≈ EMA233)"
    summary_lines.append(f"  - 长期趋势背景: {ctx_text}")

    # 短期趋势排列 (不需要格式化)
    alignment = latest_hist_row.get('alignment_signal')
    align_text = "信号不足"
    if not pd.isna(alignment):
        alignment = int(alignment)
        if alignment == 3: align_text = "完全多头 (+3)"
        elif alignment == -3: align_text = "完全空头 (-3)"
        elif alignment > 0: align_text = f"偏多头 ({alignment})"
        elif alignment < 0: align_text = f"偏空头 ({alignment})"
        else: align_text = "混合/粘合 (0)"
    summary_lines.append(f"  - 短期趋势排列 (5/13/21/55 EMA): {align_text}")

    # 趋势反转 (不需要格式化)
    reversal = latest_hist_row.get('reversal_signal', 0)
    reversal_text = "无明显信号"
    if reversal == 1: reversal_text = "**注意：潜在底部反转信号**"
    elif reversal == -1: reversal_text = "**注意：潜在顶部反转信号**"
    summary_lines.append(f"  - 趋势反转信号: {reversal_text}")

    # 评分动能 (使用 format_value)
    momentum = latest_hist_row.get('score_momentum')
    mom_formatted = format_value(momentum, '.2f') # 先格式化数字部分
    mom_text = mom_formatted
    if isinstance(clean_value(momentum), (int, float, Decimal)): # 仅当原始值是数字时添加描述
        if momentum > 0.5: mom_text += " (显著上升)"
        elif momentum > 0: mom_text += " (上升)"
        elif momentum < -0.5: mom_text += " (显著下降)"
        elif momentum < 0: mom_text += " (下降)"
        else: mom_text += " (持平)"
    summary_lines.append(f"  - 评分动能 (单期变化): {mom_text}")

    # 信号稳定性 (不需要格式化)
    stable_signal = "数据不足"
    # ... (稳定性逻辑不变) ...
    summary_lines.append(f"  - 信号稳定性 (近3期): {stable_signal}")


    # 评分波动性 (使用 format_value)
    volatility = latest_hist_row.get('score_volatility')
    vol_formatted = format_value(volatility, '.2f') # 先格式化数字部分
    vol_text = vol_formatted
    if isinstance(clean_value(volatility), (int, float, Decimal)): # 仅当原始值是数字时添加描述
        try:
            valid_vol = analysis_df['score_volatility'].dropna()
            if len(valid_vol) > VOLATILITY_WINDOW * 2:
                q75 = valid_vol.quantile(0.75)
                q25 = valid_vol.quantile(0.25)
                if volatility > q75: vol_text += " (偏高)"
                elif volatility < q25: vol_text += " (偏低)"
                else: vol_text += " (适中)"
            else: vol_text += " (历史数据不足)"
        except Exception: vol_text += " (分位数计算失败)"
    summary_lines.append(f"  - 评分波动性 ({VOLATILITY_WINDOW}期 std): {vol_text}")


    # --- T+0 信号部分 (使用 format_value) ---
    summary_lines.append(f"--- 日内 T+0 交易信号 (基于实时价格 vs 最新历史 VWAP, 长期过滤: {'启用' if t0_params.get('use_long_term_filter', False) else '禁用'}) ---")
    if not t0_params['enabled']:
        summary_lines.append("  - T+0 信号: 功能未启用")
    elif realtime_cache.get('fetch_error'):
        summary_lines.append(f"  - 实时状态: 获取实时数据失败 ({realtime_cache.get('message', 'Unknown error')})")
        summary_lines.append("  - T+0 信号: 无法判断")
    elif realtime_cache.get('realtime_price') is None:
        summary_lines.append(f"  - 实时状态: {realtime_cache.get('message', '未获取到有效实时价格')}")
        summary_lines.append("  - T+0 信号: 无法判断")
    elif realtime_cache.get('latest_vwap_used') is None or realtime_cache.get('latest_vwap_used') == 0:
         summary_lines.append(f"  - 实时价格: {format_value(realtime_cache.get('realtime_price'), '.2f')}")
         summary_lines.append(f"  - 最新历史 VWAP: 无效或为零 ({realtime_cache.get('latest_vwap_used')})")
         summary_lines.append("  - T+0 信号: 无法判断 (VWAP无效)")
    else:
        # 显示实时信息
        summary_lines.append(f"  - 实时价格: {format_value(realtime_cache.get('realtime_price'), '.2f')} (时间: {realtime_cache.get('realtime_trade_time', 'N/A')})")
        summary_lines.append(f"  - 最新历史 VWAP: {format_value(realtime_cache.get('latest_vwap_used'), '.2f')}")
        current_deviation = realtime_cache.get('current_deviation')
        summary_lines.append(f"  - 当前价格相对 VWAP 偏离度: {format_value(current_deviation, '.2%')}") # 使用百分比格式

        # 显示实时 T+0 信号
        rt_t0_signal = realtime_cache.get('current_t0_signal', 0)
        buy_dev_threshold_str = format_value(t0_params['buy_dev_threshold'], '.2%') # 格式化阈值
        sell_dev_threshold_str = format_value(t0_params['sell_dev_threshold'], '.2%') # 格式化阈值
        if rt_t0_signal == 1:
            summary_lines.append(f"  - T+0 信号: **潜在买入点** (短期趋势向好, 价<VWAP阈值 {buy_dev_threshold_str}, 长期趋势允许)")
        elif rt_t0_signal == -1:
            summary_lines.append(f"  - T+0 信号: **潜在卖出点** (短期趋势向差, 价>VWAP阈值 {sell_dev_threshold_str}, 长期趋势允许)")
        else: # rt_t0_signal == 0
            summary_lines.append("  - T+0 信号: 无或观望")
            # --- 原因判断逻辑 (保持不变) ---
            hist_alignment = latest_hist_row.get('alignment_signal')
            hist_long_term_ctx = latest_hist_row.get('long_term_context')
            reason = "不满足信号条件"
            if pd.isna(hist_alignment):
                reason = "短期排列信号不足"
            elif current_deviation is None or pd.isna(current_deviation):
                 reason = "实时价格偏离度未知"
            elif hist_alignment >= 1 and current_deviation >= t0_params['buy_dev_threshold']:
                reason = "价格未低于买入阈值"
            elif hist_alignment <= -1 and current_deviation <= t0_params['sell_dev_threshold']:
                reason = "价格未高于卖出阈值"
            elif t0_params['use_long_term_filter']:
                 is_long_term_ctx_valid_for_buy = pd.isna(hist_long_term_ctx) or hist_long_term_ctx >= 0
                 is_long_term_ctx_valid_for_sell = pd.isna(hist_long_term_ctx) or hist_long_term_ctx <= 0
                 if hist_alignment >= 1 and not is_long_term_ctx_valid_for_buy:
                     reason = "长期趋势不允许买入"
                 elif hist_alignment <= -1 and not is_long_term_ctx_valid_for_sell:
                     reason = "长期趋势不允许卖出"
            summary_lines.append(f"    - 原因: {reason}")

    return "\n".join(summary_lines)

def _prepare_latest_historical_cache(latest_row: pd.Series) -> Dict[str, Any]:
    """准备用于缓存的最新历史数据字典。"""
    ema_233_col = f'ema_score_{LONG_TERM_EMA_PERIOD}'
    return {
        'trade_time': latest_row.name.isoformat() if isinstance(latest_row.name, pd.Timestamp) else str(latest_row.name),
        'score': clean_value(latest_row.get('score')),
        'close_price': clean_value(latest_row.get('close')),
        'vwap': clean_value(latest_row.get('vwap')),
        'ema_score_5': clean_value(latest_row.get('ema_score_5')),
        'ema_score_13': clean_value(latest_row.get('ema_score_13')),
        'ema_score_21': clean_value(latest_row.get('ema_score_21')),
        'ema_score_55': clean_value(latest_row.get('ema_score_55')),
        'ema_score_233': clean_value(latest_row.get(ema_233_col)),
        'alignment_signal': clean_value(latest_row.get('alignment_signal')),
        'ema_strength_13_55': clean_value(latest_row.get('ema_strength_13_55')),
        'score_momentum': clean_value(latest_row.get('score_momentum')),
        'score_volatility': clean_value(latest_row.get('score_volatility')),
        'long_term_context': clean_value(latest_row.get('long_term_context')),
        'reversal_signal': clean_value(latest_row.get('reversal_signal')),
        'price_vwap_deviation': clean_value(latest_row.get('price_vwap_deviation')),
        't0_signal_hist': clean_value(latest_row.get('t0_signal')), # 历史 T+0 信号
    }

@sync_to_async # 恢复使用 sync_to_async
def _save_single_analysis_to_db(stock_instance: StockInfo, data_row: pd.Series, level: str) -> Tuple[bool, bool]:
    """(修正版) 使用 ORM 的 update_or_create 保存单行数据。返回 (success, created)。"""
    trade_time_aware = data_row.name
    stock_code = stock_instance.stock_code # 用于日志

    if not isinstance(trade_time_aware, pd.Timestamp) or trade_time_aware.tzinfo is None:
        logger.warning(f"[{stock_code}] 保存数据库时时间戳无效或无时区: {trade_time_aware}, 跳过。")
        return False, False # success=False, created=False

    ema_233_col = f'ema_score_{LONG_TERM_EMA_PERIOD}'
    defaults = {
        'score': clean_value(data_row.get('score')),
        'close_price': clean_value(data_row.get('close')),
        'vwap': clean_value(data_row.get('vwap')),
        'ema_score_5': clean_value(data_row.get('ema_score_5')),
        'ema_score_13': clean_value(data_row.get('ema_score_13')),
        'ema_score_21': clean_value(data_row.get('ema_score_21')),
        'ema_score_55': clean_value(data_row.get('ema_score_55')),
        'ema_score_233': clean_value(data_row.get(ema_233_col)),
        'alignment_signal': clean_value(data_row.get('alignment_signal')),
        'ema_strength_13_55': clean_value(data_row.get('ema_strength_13_55')),
        'score_momentum': clean_value(data_row.get('score_momentum')),
        'score_volatility': clean_value(data_row.get('score_volatility')),
        'long_term_context': clean_value(data_row.get('long_term_context')),
        'reversal_signal': clean_value(data_row.get('reversal_signal')),
        'price_vwap_deviation': clean_value(data_row.get('price_vwap_deviation')),
        't0_signal': clean_value(data_row.get('t0_signal')),
    }
    # 移除值为 None 的键
    defaults = {k: v for k, v in defaults.items() if v is not None}

    try:
        # 使用同步的 update_or_create，因为函数被 sync_to_async 包装
        obj, created = StockScoreAnalysis.objects.update_or_create(
            stock=stock_instance,
            trade_time=trade_time_aware,
            time_level=level,
            defaults=defaults
        )
        return True, created # success=True
    except Exception as db_err:
        logger.error(f"[{stock_code}] 保存数据到数据库时出错 (时间: {trade_time_aware}, 级别: {level}): {db_err}", exc_info=True)
        return False, False # success=False

async def _save_analysis_results_db(stock_code: str, df: pd.DataFrame, level: str):
    """(修正调用方式) 筛选有效数据并并发保存到数据库。"""
    if StockScoreAnalysis is None or StockInfo is None:
        logger.warning(f"[{stock_code}] 数据库模型未加载，跳过数据库保存。")
        return
    if level == 'unknown':
        logger.warning(f"[{stock_code}] 分析时间级别未知，跳过数据库保存。")
        return

    stock_instance: Optional[StockInfo] = None
    try:
        # 仍然可以使用 aget 获取 StockInfo 实例 (Django 4.1+)
        stock_instance = await StockInfo.objects.aget(stock_code=stock_code)
    except StockInfo.DoesNotExist:
        logger.error(f"[{stock_code}] 数据库中未找到股票 {stock_code}，无法保存分析结果。")
        return
    except AttributeError: # 处理 Django < 4.1 的情况 或 aget 不可用
         try:
             stock_instance = await sync_to_async(StockInfo.objects.get)(stock_code=stock_code)
         except StockInfo.DoesNotExist:
             logger.error(f"[{stock_code}] 数据库中未找到股票 {stock_code} (sync fallback)，无法保存分析结果。")
             return
         except Exception as e_sync:
             logger.error(f"[{stock_code}] 获取 StockInfo 时出错 (sync fallback): {e_sync}", exc_info=True)
             return
    except Exception as e:
        logger.error(f"[{stock_code}] 获取 StockInfo 时出错: {e}", exc_info=True)
        return

    # 筛选包含有效核心信号的行进行保存
    valid_rows = df.dropna(subset=['alignment_signal'])
    rows_to_save = valid_rows.iloc[-DB_SAVE_RECENT_N:]

    if rows_to_save.empty:
        logger.info(f"[{stock_code}] 没有找到有效的分析记录用于保存到数据库。")
        return

    logger.info(f"[{stock_code}] 准备保存 {len(rows_to_save)} 条最近有效分析记录到数据库 (级别: {level})...")
    # tasks 列表包含对 sync_to_async 包装后的函数的调用，这些调用返回 awaitable 对象
    tasks = [_save_single_analysis_to_db(stock_instance, row, level) for _, row in rows_to_save.iterrows()]
    results = await asyncio.gather(*tasks, return_exceptions=True) # 正确 await 调用

    # 统计结果 (保持不变)
    success_count = sum(1 for r in results if isinstance(r, tuple) and r[0] is True)
    created_count = sum(1 for r in results if isinstance(r, tuple) and r[0] is True and r[1] is True)
    failed_count = len(results) - success_count
    logger.info(f"[{stock_code}] 数据库保存完成: 成功 {success_count} 条 (新增 {created_count} 条), 失败 {failed_count} 条。")


async def _save_analysis_results_db(stock_code: str, df: pd.DataFrame, level: str):
    """筛选有效数据并并发保存到数据库。"""
    if StockScoreAnalysis is None or StockInfo is None:
        logger.warning(f"[{stock_code}] 数据库模型未加载，跳过数据库保存。")
        return
    if level == 'unknown':
        logger.warning(f"[{stock_code}] 分析时间级别未知，跳过数据库保存。")
        return

    try:
        # Django 5.0 支持 aget
        stock_instance = await StockInfo.objects.aget(stock_code=stock_code)
    except StockInfo.DoesNotExist:
        logger.error(f"[{stock_code}] 数据库中未找到股票 {stock_code}，无法保存分析结果。")
        return
    except Exception as e:
        logger.error(f"[{stock_code}] 获取 StockInfo 时出错: {e}", exc_info=True)
        return

    # 筛选包含有效核心信号的行进行保存
    valid_rows = df.dropna(subset=['alignment_signal']) # 或者选择更关键的指标
    rows_to_save = valid_rows.iloc[-DB_SAVE_RECENT_N:]

    if rows_to_save.empty:
        logger.info(f"[{stock_code}] 没有找到有效的分析记录用于保存到数据库。")
        return

    logger.info(f"[{stock_code}] 准备保存 {len(rows_to_save)} 条最近有效分析记录到数据库 (级别: {level})...")
    tasks = [_save_single_analysis_to_db(stock_instance, row, level) for _, row in rows_to_save.iterrows()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 统计结果
    success_count = sum(1 for r in results if isinstance(r, tuple) and r[0] is True)
    created_count = sum(1 for r in results if isinstance(r, tuple) and r[0] is True and r[1] is True)
    failed_count = len(results) - success_count

    logger.info(f"[{stock_code}] 数据库保存完成: 成功 {success_count} 条 (新增 {created_count} 条), 失败 {failed_count} 条。")


async def _cache_analysis_results_redis(stock_code: str, level: str, latest_hist_cache: Dict, realtime_cache: Dict, summary: str):
    """将分析结果结构化缓存到 Redis。"""
    if not CacheManager:
        logger.warning(f"[{stock_code}] CacheManager 未初始化，跳过 Redis 缓存。")
        return

    logger.info(f"[{stock_code}] 开始缓存最新分析状态到 Redis...")
    try:
        cache_manager = CacheManager()
        cache_key = cache_manager.generate_key('analysis', 'latest', stock_code, level) # 建议 key 包含 level
        cache_data = {
            'stock_code': stock_code,
            'analysis_time_level': level,
            'latest_historical_data': latest_hist_cache,
            'realtime_data': realtime_cache,
            'summary_text': summary,
            'cache_timestamp': timezone.now().isoformat()
        }
        success = await cache_manager.set(cache_key, cache_data, timeout=REDIS_CACHE_TIMEOUT)
        if success:
            logger.info(f"[{stock_code}] 最新分析状态已缓存至 Redis (Key: {cache_key})。")
        else:
            logger.warning(f"[{stock_code}] 缓存最新分析状态至 Redis 失败 (Key: {cache_key})。")
    except ConnectionError as e:
        logger.error(f"[{stock_code}] 缓存时 Redis 连接错误: {e}")
    except Exception as e:
        logger.error(f"[{stock_code}] 缓存分析结果至 Redis 时发生未知错误: {e}", exc_info=True)


# --- 主分析函数 (重构后) ---
async def analyze_score_trend(stock_code: str,
    score_price_vwap_df: pd.DataFrame,
    t0_params_override: Optional[Dict[str, Any]] = None, # T+0 参数覆盖
    save_to_db: bool = True,
    cache_results: bool = True,
    analysis_time_level_override: Optional[str] = None # 覆盖自动检测的级别
    ) -> Optional[pd.DataFrame]:
    """
    (重构版) 分析策略评分和价格趋势，生成 T+0 信号，保存并缓存结果。

    Args:
        stock_code (str): 股票代码。
        score_price_vwap_df (pd.DataFrame): 输入数据 (score, close, [vwap])。
        t0_params_override (Optional[Dict[str, Any]]): 覆盖默认 T+0 参数。
        save_to_db (bool): 是否保存到数据库。
        cache_results (bool): 是否缓存到 Redis。
        analysis_time_level_override (Optional[str]): 强制指定分析级别。

    Returns:
        Optional[pd.DataFrame]: 包含完整分析结果的 DataFrame，或 None。
    """
    # --- 0. 参数处理 ---
    t0_params = {**DEFAULT_T0_PARAMS, **(t0_params_override or {})}
    realtime_dao = StockRealtimeDAO() # 初始化 DAO

    # --- 1. 输入验证 ---
    t0_enabled = t0_params['enabled']
    if not _validate_input_data(stock_code, score_price_vwap_df, t0_enabled):
        return None
    # 如果验证通过但缺少 vwap，则禁用 T+0
    if t0_enabled and 'vwap' not in score_price_vwap_df.columns:
        logger.warning(f"[{stock_code}] 输入 DataFrame 缺少 'vwap' 列，T+0 信号功能已禁用。")
        t0_params['enabled'] = False
        t0_enabled = False

    # --- 2. 复制数据 & 确定分析级别 ---
    analysis_df = score_price_vwap_df.copy()
    analysis_level = _determine_analysis_level(stock_code, analysis_df, analysis_time_level_override)
    logger.info(f"[{stock_code}] 开始分析 (级别: {analysis_level}, DB: {save_to_db}, Cache: {cache_results})...")

    # --- 3. 时区标准化 ---
    analysis_df = _standardize_timezone(stock_code, analysis_df)
    if analysis_df is None: return None # 时区处理失败

    # --- 4. 数据量检查 (针对长周期 EMA) ---
    min_required_data = LONG_TERM_EMA_PERIOD + 15
    if len(analysis_df) < min_required_data:
        logger.warning(f"[{stock_code}] 数据点 ({len(analysis_df)}) 不足 {min_required_data}，长周期指标 ({LONG_TERM_EMA_PERIOD} EMA) 可能不准确或为 NaN。")

    # --- 5. 计算核心指标 ---
    analysis_df = _calculate_indicators(stock_code, analysis_df)

    # --- 6. 计算历史 T+0 信号 ---
    analysis_df = _calculate_historical_t0(stock_code, analysis_df, t0_params)

    # --- 7. 获取实时数据并计算实时 T+0 ---
    latest_hist_row = analysis_df.iloc[-1]
    realtime_cache = await _fetch_and_process_realtime(stock_code, realtime_dao, latest_hist_row, t0_params)

    # --- 8. 生成分析摘要 ---
    summary = _generate_summary(stock_code, latest_hist_row, realtime_cache, analysis_df, t0_params, analysis_level)
    print("\n" + "="*30 + " 评分与价格趋势分析摘要 " + "="*30)
    print(summary)
    print("="* (60 + len(" 评分与价格趋势分析摘要 "))) # 动态调整分隔线长度

    # --- 9. 准备缓存数据 ---
    latest_hist_cache = _prepare_latest_historical_cache(latest_hist_row)

    # --- 10. 保存到数据库 (异步) ---
    if save_to_db:
        # 使用 asyncio.create_task 在后台执行，不阻塞后续缓存操作
        # 注意：如果后续逻辑依赖数据库操作完成，则需要 await
        asyncio.create_task(_save_analysis_results_db(stock_code, analysis_df, analysis_level))
        # 或者如果需要等待： await _save_analysis_results_db(stock_code, analysis_df, analysis_level)
    else:
        logger.info(f"[{stock_code}] 跳过数据库保存 (save_to_db=False)。")


    # --- 11. 缓存到 Redis (异步) ---
    if cache_results:
        # 使用 asyncio.create_task 在后台执行
        asyncio.create_task(_cache_analysis_results_redis(stock_code, analysis_level, latest_hist_cache, realtime_cache, summary))
        # 或者如果需要等待： await _cache_analysis_results_redis(...)
    else:
        logger.info(f"[{stock_code}] 跳过 Redis 缓存 (cache_results=False)。")

    # --- 12. 返回结果 ---
    logger.info(f"[{stock_code}] 分析流程完成。")
    return analysis_df

# --- test_strategy_scores 函数 (保持不变，但调用重构后的 analyze_score_trend) ---
async def test_strategy_scores(stock_code: str, time_level_for_analysis: str = '5'):
    """
    测试指定股票代码的策略评分生成过程，并进行趋势和 T+0 分析。
    (调用重构后的 analyze_score_trend)

    Args:
        stock_code (str): 股票代码。
        time_level_for_analysis (str): 用于 T+0 分析和选取 close/vwap 列的时间周期。
    """
    local_tz = None
    local_tz_name = "系统默认"
    try:
        local_tz = tzlocal.get_localzone()
        local_tz_name = str(local_tz)
        logger.info(f"检测到本地时区: {local_tz_name}")
    except Exception as tz_e:
        logger.warning(f"获取本地时区时出错: {tz_e}. 时间将不会转换。")

    # 1. 初始化服务和 DAO
    indicator_service = IndicatorService()
    stock_basic_dao = StockBasicDAO()

    # 2. 定义策略参数 (建议从配置加载)
    # TODO: 从配置文件或数据库加载策略参数
    strategy_params: Dict[str, Any] = {
        'rsi_period': 14, 'rsi_oversold': 25, 'rsi_overbought': 75, 'rsi_extreme_oversold': 20, 'rsi_extreme_overbought': 80,
        'kdj_period_k': 12, 'kdj_period_d': 3, 'kdj_period_j': 3, 'kdj_oversold': 20, 'kdj_overbought': 80,
        'boll_period': 20, 'boll_std_dev': 2,
        'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
        'cci_period': 14, 'cci_threshold': 100, 'cci_extreme_threshold': 200,
        'mfi_period': 14, 'mfi_oversold': 20, 'mfi_overbought': 80, 'mfi_extreme_oversold': 10, 'mfi_extreme_overbought': 90,
        'roc_period': 12,
        'dmi_period': 14, 'adx_threshold': 20, 'adx_strong_threshold': 30,
        'sar_step': 0.02, 'sar_max': 0.2,
        'amount_ma_period': 20, 'obv_ma_period': 10, 'cmf_period': 20,
        'ema_period': 13,
        'weights': {'5': 0.1, '15': 0.4, '30': 0.3, '60': 0.2},
        'volume_tf': '15', # VWAP 通常基于日内分钟线计算，这里假设 '15' 分钟 VWAP
        'volume_confirmation': True, 'volume_confirm_boost': 1.1, 'volume_fail_penalty': 0.8, 'divergence_penalty': 0.3,
        'check_bearish_divergence': True, 'divergence_price_period': 5,
        'divergence_threshold_cmf': -0.05, 'divergence_threshold_mfi': 40,
    }
    strategy_instance = MacdRsiKdjBollEnhancedStrategy(params=strategy_params)

    # 3. 确定所需时间周期
    strategy_timeframes = strategy_instance.timeframes
    all_required_timeframes = set(strategy_timeframes)
    all_required_timeframes.add(time_level_for_analysis) # 分析用的价格周期
    volume_tf = strategy_params.get('volume_tf', '15') # 获取 VWAP 的周期
    all_required_timeframes.add(volume_tf)
    timeframes_list = sorted(list(all_required_timeframes), key=lambda x: int(x) if x.isdigit() else float('inf')) # 排序

    stock = await stock_basic_dao.get_stock_by_code(stock_code)
    if not stock:
        logger.error(f"无法找到股票信息: {stock_code}")
        return

    # 4. 准备策略数据 (增加数据量以计算长周期 EMA)
    # TODO: 使 limit_count 可配置或动态计算
    limit_count = LONG_TERM_EMA_PERIOD + 100 # 确保有足够数据计算最长 EMA
    logger.info(f"[{stock}] 正在准备统一策略数据 (周期: {timeframes_list}, 限制: {limit_count})...")
    strategy_df: Optional[pd.DataFrame] = await indicator_service.prepare_strategy_dataframe(
        stock_code=stock_code,
        timeframes=timeframes_list,
        strategy_params=strategy_params,
        limit_per_tf=limit_count # 传递限制
    )

    if strategy_df is None or strategy_df.empty:
        logger.error(f"[{stock}] 统一策略数据准备失败或为空。")
        return
    logger.info(f"[{stock}] 统一策略数据准备完成，形状: {strategy_df.shape}")
    # logger.debug(f"[{stock}] strategy_df columns: {strategy_df.columns.tolist()}")

    # 5. 生成评分
    logger.info(f"[{stock}] 正在生成策略评分...")
    scores: Optional[pd.Series] = None
    try:
        scores = strategy_instance.run(strategy_df)
        # intermediate_data = strategy_instance.get_intermediate_data() # 如果需要中间数据
        logger.info(f"[{stock}] 策略评分生成完成。")

        if scores is not None and not scores.empty:
            scores_display = scores.copy()
            if local_tz and isinstance(scores_display.index, pd.DatetimeIndex):
                try:
                    scores_display.index = scores_display.index.tz_convert(local_tz)
                except TypeError: # 如果已经是 naive datetime
                    pass

            print(f"\n[{stock}] 最新的评分 (最后10条，时间：{local_tz_name}):")
            print(scores_display.tail(10).round(2))
            nan_count = scores.isna().sum()
            if nan_count > 0: print(f"\n警告: 生成的评分中包含 {nan_count} 个 NaN 值。")

            # 6. 准备分析输入 DataFrame
            logger.info(f"[{stock}] 准备分析输入数据...")
            price_col = f'close_{time_level_for_analysis}'
            vwap_col = f'vwap_{volume_tf}' # 使用 volume_tf 获取 VWAP 列名

            analysis_input_data = {'score': scores}
            missing_analysis_cols = []

            if price_col in strategy_df.columns:
                analysis_input_data['close'] = strategy_df[price_col]
            else:
                missing_analysis_cols.append(price_col)
                analysis_input_data['close'] = pd.Series(dtype=float) # 添加空列以保持结构

            if vwap_col in strategy_df.columns:
                analysis_input_data['vwap'] = strategy_df[vwap_col]
            else:
                missing_analysis_cols.append(vwap_col)
                analysis_input_data['vwap'] = pd.Series(dtype=float) # 添加空列

            if missing_analysis_cols:
                 logger.warning(f"[{stock}] 策略 DataFrame 中缺少分析所需的列: {missing_analysis_cols}。分析可能不完整或 T+0 会被禁用。")

            # 合并到一个 DataFrame，以 score 的索引为基准
            score_price_vwap_df = pd.DataFrame(index=scores.index)
            score_price_vwap_df['score'] = scores
            # 使用 reindex 确保索引对齐，填充缺失值为 NaN
            if 'close' in analysis_input_data:
                score_price_vwap_df['close'] = analysis_input_data['close'].reindex(scores.index)
            if 'vwap' in analysis_input_data:
                score_price_vwap_df['vwap'] = analysis_input_data['vwap'].reindex(scores.index)

            # 删除 score 或 close 为 NaN 的行，这些是分析的基础
            initial_len = len(score_price_vwap_df)
            score_price_vwap_df.dropna(subset=['score', 'close'], how='any', inplace=True)
            dropped_count = initial_len - len(score_price_vwap_df)
            if dropped_count > 0:
                logger.info(f"[{stock}] 移除了 {dropped_count} 行缺少 score 或 close 的数据。")

            if not score_price_vwap_df.empty:
                logger.info(f"[{stock}] 分析输入数据准备完成 (数据条数: {len(score_price_vwap_df)})，开始调用 analyze_score_trend...")
                # 7. 调用重构后的分析函数
                # TODO: 从配置加载 T+0 参数
                t0_settings_override = {
                    # 'enabled': True, # 可以覆盖默认值
                    # 'buy_dev_threshold': -0.004,
                    # 'sell_dev_threshold': 0.006,
                }
                analysis_result_df = await analyze_score_trend(
                    stock_code=str(stock_code),
                    score_price_vwap_df=score_price_vwap_df,
                    t0_params_override=t0_settings_override,
                    save_to_db=True, # 控制是否保存
                    cache_results=True, # 控制是否缓存
                    analysis_time_level_override=time_level_for_analysis # 明确传递分析级别
                )
                # analysis_result_df 可以在这里进一步使用
                if analysis_result_df is not None:
                     logger.info(f"[{stock}] analyze_score_trend 执行完毕，返回 DataFrame 形状: {analysis_result_df.shape}")
                else:
                     logger.warning(f"[{stock}] analyze_score_trend 返回 None。")

            else:
                logger.warning(f"[{stock}] 准备用于分析的数据为空或失败，跳过趋势分析。")

        else:
            logger.error(f"\n[{stock}] 未能获取有效的评分结果。")

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



# --- 更新：读取并展示缓存数据的函数 (使用 CacheManager) ---
async def display_cached_analysis(stock_code: str):
    """
    从 Redis 读取指定股票的最新分析缓存数据 (使用 CacheManager) 并格式化输出。
    """
    if CacheManager: # 检查 CacheManager 是否可用
        cache_manager = CacheManager()
        # 使用与存储时相同的逻辑生成键
        cache_key = cache_manager.generate_key('analysis', 'latest', stock_code)
        logger.info(f"尝试从 Redis 读取缓存数据 (使用 CacheManager, Key: {cache_key})...")

        try:
            # 调用 CacheManager 的 get 方法，它处理反序列化 (umsgpack)
            cached_data = await cache_manager.get(cache_key)

            if cached_data:
                # get 方法成功时返回反序列化后的 Python 对象 (通常是字典)
                if isinstance(cached_data, dict):
                    logger.info(f"成功读取并反序列化缓存数据 (缓存时间: {cached_data.get('cache_timestamp', 'N/A')})。")

                    # --- 输出方式一：直接打印缓存的摘要文本 ---
                    # print("\n" + "="*30 + f" {stock_code} 缓存分析摘要 (来自 Redis - CacheManager) " + "="*30)
                    # print(cached_data.get('summary_text', '缓存中未找到摘要文本。'))
                    # print("="* (62 + len(stock_code) + len(" 缓存分析摘要 (来自 Redis - CacheManager) ")))

                    # --- 输出方式二：根据结构化数据动态生成格式化输出 (可选) ---
                    print("\n" + "="*30 + f" {stock_code} 结构化缓存数据详情 " + "="*30)
                    hist_data = cached_data.get('latest_historical_data', {})
                    rt_data = cached_data.get('realtime_data', {})
                    print(f"分析级别: {cached_data.get('analysis_time_level', 'N/A')}")
                    # ... 打印其他字段 ...
                    print("="* (62 + len(stock_code) + len(" 结构化缓存数据详情 ")))
                else:
                    logger.error(f"[{stock_code}] 从 CacheManager 获取的数据类型不是预期的字典: type={type(cached_data)}")
                    print(f"\n错误：股票 {stock_code} 的缓存数据格式不正确。")

            else:
                # get 方法返回 None 表示缓存未命中
                logger.warning(f"[{stock_code}] 在 Redis 中未找到缓存数据 (使用 CacheManager, Key: {cache_key})。")
                print(f"\n未找到股票 {stock_code} 的缓存分析数据。")

        except ConnectionError as e:
             logger.error(f"[{stock_code}] 读取缓存时 Redis 连接错误: {e}")
             print(f"\n错误：读取股票 {stock_code} 的缓存时连接 Redis 失败。")
        except Exception as e:
             logger.error(f"[{stock_code}] 处理 Redis 缓存数据时发生错误: {e}", exc_info=True)
             print(f"\n错误：处理股票 {stock_code} 的缓存数据时发生错误。")