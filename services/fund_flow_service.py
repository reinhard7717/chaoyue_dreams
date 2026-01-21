# services/fund_flow_service.py

import asyncio
import logging
from django.utils import timezone
import pandas as pd
import numpy as np
from datetime import timedelta, datetime, time
from typing import Tuple
import numba
from asgiref.sync import sync_to_async
from stock_models.stock_basic import StockInfo
from stock_models.time_trade import StockDailyBasic
from stock_models.advanced_metrics import BaseAdvancedFundFlowMetrics
from utils.model_helpers import (
    get_advanced_fund_flow_metrics_model_by_code,
    get_daily_data_model_by_code,
)

logger = logging.getLogger('services')

@numba.njit(cache=True)
def _numba_calculate_attribution_modifiers(
    vol_shares_arr: np.ndarray,
    vol_ma_arr: np.ndarray,
    price_range_arr: np.ndarray,
    range_ma_arr: np.ndarray,
    minute_vwap_arr: np.ndarray,
    daily_vwap: float,
    momentum_modifier_raw_arr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    【Numba优化版】计算归因权重中的各种修饰符。
    """
    impulse_modifier = np.ones_like(vol_shares_arr, dtype=np.float64)
    lg_buy_modifier = np.ones_like(vol_shares_arr, dtype=np.float64)
    lg_sell_modifier = np.ones_like(vol_shares_arr, dtype=np.float64)
    md_buy_modifier = np.ones_like(vol_shares_arr, dtype=np.float64)
    md_sell_modifier = np.ones_like(vol_shares_arr, dtype=np.float64)
    # impulse_modifier
    for i in range(len(vol_shares_arr)):
        if vol_ma_arr[i] > 1e-9 and range_ma_arr[i] > 1e-9:
            impulse_modifier[i] = (vol_shares_arr[i] / vol_ma_arr[i]) * (price_range_arr[i] / range_ma_arr[i])
    impulse_modifier = np.clip(impulse_modifier, 0, 10)
    # lg_buy_modifier, lg_sell_modifier
    if not np.isnan(daily_vwap):
        for i in range(len(minute_vwap_arr)):
            if daily_vwap > 1e-9:
                vwap_deviation = (minute_vwap_arr[i] - daily_vwap) / daily_vwap
                lg_buy_modifier[i] = np.exp(-np.maximum(0, vwap_deviation) * 5)
                lg_sell_modifier[i] = np.exp(np.minimum(0, vwap_deviation) * 5)
    # md_buy_modifier, md_sell_modifier
    for i in range(len(momentum_modifier_raw_arr)):
        md_buy_modifier[i] = np.exp(momentum_modifier_raw_arr[i] * 50)
        md_sell_modifier[i] = np.exp(-momentum_modifier_raw_arr[i] * 50)
    return impulse_modifier, lg_buy_modifier, lg_sell_modifier, md_buy_modifier, md_sell_modifier

@numba.njit(cache=True)
def _numba_calculate_level5_ofi_components(
    buy_volumes: np.ndarray, sell_volumes: np.ndarray,
    buy_prices: np.ndarray, sell_prices: np.ndarray,
    q_threshold: float, weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int, int, float, float, int]:
    """
    Numba优化版：计算Level 5订单流的组件。
    输入：
        buy_volumes (np.ndarray): 形状为 (N, 5) 的买盘量数组，N为时间步数，5为档位。
        sell_volumes (np.ndarray): 形状为 (N, 5) 的卖盘量数组。
        buy_prices (np.ndarray): 形状为 (N, 5) 的买盘价数组。
        sell_prices (np.ndarray): 形状为 (N, 5) 的卖盘价数组。
        q_threshold (float): 主力订单量阈值。
        weights (np.ndarray): 形状为 (5,) 的档位权重数组。
    返回：
        Tuple:
            - main_force_ofi_snapshots (np.ndarray): 主力订单流快照。
            - retail_ofi_snapshots (np.ndarray): 散户订单流快照。
            - mf_replenishment_bid_events (int): 主力买盘补单事件计数。
            - mf_replenishment_ask_events (int): 主力卖盘补单事件计数。
            - mf_cancellation_volume (float): 主力撤单量。
            - mf_total_posted_volume (float): 主力总挂单量。
            - num_rows (int): 处理的行数。
    """
    num_rows = buy_volumes.shape[0]
    main_force_ofi_snapshots = np.zeros(num_rows, dtype=np.float64)
    retail_ofi_snapshots = np.zeros(num_rows, dtype=np.float64)
    mf_replenishment_bid_events = 0
    mf_replenishment_ask_events = 0
    mf_cancellation_volume = 0.0
    mf_total_posted_volume = 0.0
    # prev_row_data stores [buy_volume1..5, sell_volume1..5, buy_price1..5, sell_price1..5]
    # Initialize with zeros for the first row's 'prev_row' logic
    prev_buy_volumes = np.zeros(5, dtype=np.float64)
    prev_sell_volumes = np.zeros(5, dtype=np.float64)
    prev_buy_prices = np.zeros(5, dtype=np.float64)
    prev_sell_prices = np.zeros(5, dtype=np.float64)
    for i in range(num_rows):
        main_force_bid_pressure = 0.0
        main_force_ask_pressure = 0.0
        retail_bid_pressure = 0.0
        retail_ask_pressure = 0.0
        for j in range(5): # Loop through 5 levels
            buy_vol = buy_volumes[i, j]
            sell_vol = sell_volumes[i, j]
            buy_price = buy_prices[i, j]
            sell_price = sell_prices[i, j]
            if buy_vol > 0 and buy_price > 0:
                if buy_vol >= q_threshold:
                    main_force_bid_pressure += buy_vol * weights[j]
                    mf_total_posted_volume += buy_vol
                    if i > 0 and prev_buy_volumes[j] > 0 and buy_vol > prev_buy_volumes[j] and buy_price == prev_buy_prices[j]:
                        mf_replenishment_bid_events += 1
                else:
                    retail_bid_pressure += buy_vol * weights[j]
            if sell_vol > 0 and sell_price > 0:
                if sell_vol >= q_threshold:
                    main_force_ask_pressure += sell_vol * weights[j]
                    mf_total_posted_volume += sell_vol
                    if i > 0 and prev_sell_volumes[j] > 0 and sell_vol > prev_sell_volumes[j] and sell_price == prev_sell_prices[j]:
                        mf_replenishment_ask_events += 1
                else:
                    retail_ask_pressure += sell_vol * weights[j]
            # Cancellation logic
            if i > 0:
                if prev_buy_volumes[j] >= q_threshold and buy_vol < q_threshold and buy_price == prev_buy_prices[j]:
                    mf_cancellation_volume += prev_buy_volumes[j] - buy_vol
                if prev_sell_volumes[j] >= q_threshold and sell_vol < q_threshold and sell_price == prev_sell_prices[j]:
                    mf_cancellation_volume += prev_sell_volumes[j] - sell_vol
        total_mf_pressure = main_force_bid_pressure + main_force_ask_pressure
        if total_mf_pressure > 0:
            main_force_ofi_snapshots[i] = (main_force_bid_pressure - main_force_ask_pressure) / total_mf_pressure
        else:
            main_force_ofi_snapshots[i] = 0.0
        total_retail_pressure = retail_bid_pressure + retail_ask_pressure
        if total_retail_pressure > 0:
            retail_ofi_snapshots[i] = (retail_bid_pressure - retail_ask_pressure) / total_retail_pressure
        else:
            retail_ofi_snapshots[i] = 0.0
        # Update prev_row_data for the next iteration
        for j in range(5):
            prev_buy_volumes[j] = buy_volumes[i, j]
            prev_sell_volumes[j] = sell_volumes[i, j]
            prev_buy_prices[j] = buy_prices[i, j]
            prev_sell_prices[j] = sell_prices[i, j]
    return (main_force_ofi_snapshots, retail_ofi_snapshots,
            mf_replenishment_bid_events, mf_replenishment_ask_events,
            mf_cancellation_volume, mf_total_posted_volume, num_rows)

@numba.njit(cache=True)
def _numba_calculate_retail_fomo_panic_scores_enhanced(
    prices: np.ndarray, volumes: np.ndarray, amounts: np.ndarray, types: np.ndarray,
    sell_price1s: np.ndarray, buy_price1s: np.ndarray,
    is_new_highs: np.ndarray, is_new_lows: np.ndarray,
    is_retail: np.ndarray, aggressive_buy: np.ndarray, aggressive_sell: np.ndarray,
    price_acceleration: np.ndarray, volume_zscore: np.ndarray,
    atr: float, cost_mf_sell: float, cost_mf_buy: float
) -> Tuple[float, float, float, float, float, float, int, int]:
    """
    增强版Numba函数：精细化计算零售FOMO和恐慌分数，考虑多维度市场微观结构。
    输入参数：
        prices: 成交价格数组
        volumes: 成交量数组
        amounts: 成交金额数组
        types: 交易类型数组 (1=买入, -1=卖出, 0=中性)
        sell_price1s: 卖一价数组
        buy_price1s: 买一价数组
        is_new_highs: 是否为新高数组
        is_new_lows: 是否为新低数组
        is_retail: 是否为零售交易数组
        aggressive_buy: 是否主动买入数组
        aggressive_sell: 是否主动卖出数组
        price_acceleration: 价格加速度数组
        volume_zscore: 成交量z-score数组
        atr: 平均真实波幅
        cost_mf_sell: 主力卖出成本
        cost_mf_buy: 主力买入成本
    返回：
        total_weighted_fomo_score: FOMO加权总分
        total_fomo_volume: FOMO总成交量
        total_fomo_amount: FOMO总成交金额
        total_weighted_panic_score: 恐慌加权总分
        total_panic_volume: 恐慌总成交量
        total_panic_amount: 恐慌总成交金额
        fomo_count: FOMO事件数量
        panic_count: 恐慌事件数量
    """
    total_weighted_fomo_score = 0.0
    total_fomo_volume = 0.0
    total_fomo_amount = 0.0
    total_weighted_panic_score = 0.0
    total_panic_volume = 0.0
    total_panic_amount = 0.0
    fomo_count = 0
    panic_count = 0
    
    num_trades = prices.shape[0]
    
    # 有效性检查
    if atr <= 0 or num_trades == 0:
        return total_weighted_fomo_score, total_fomo_volume, total_fomo_amount, \
               total_weighted_panic_score, total_panic_volume, total_panic_amount, \
               fomo_count, panic_count
    
    # 计算参考价格（如果主力成本无效，使用滚动中位数作为替代）
    if np.isnan(cost_mf_sell) or cost_mf_sell <= 0:
        # 使用前100笔交易的中位数价格作为卖出成本参考
        window_size = min(100, num_trades)
        cost_mf_sell = np.median(prices[:window_size]) if window_size > 0 else np.nan
    
    if np.isnan(cost_mf_buy) or cost_mf_buy <= 0:
        window_size = min(100, num_trades)
        cost_mf_buy = np.median(prices[:window_size]) if window_size > 0 else np.nan
    
    # 计算FOMO分数（零售在创新高时买入）
    for i in range(num_trades):
        # 检查是否为零售交易且为买入
        if is_retail[i] and types[i] == 1:
            # 检查是否创下新高
            if is_new_highs[i]:
                # 基础信息
                fomo_volume = volumes[i]
                fomo_amount = amounts[i]
                fomo_price = prices[i]
                
                # 1. 成本溢价成分：相对于主力卖出成本的溢价
                cost_premium_component = 0.0
                if not np.isnan(cost_mf_sell) and cost_mf_sell > 0:
                    cost_premium_component = (fomo_price - cost_mf_sell) / atr
                
                # 2. 攻击性成分：主动买入的惩罚
                aggression_component = 1.0
                if aggressive_buy[i]:
                    aggression_component = 1.5  # 主动买入显示更强的FOMO
                else:
                    aggression_component = 0.7  # 被动买入显示较弱的FOMO
                
                # 3. 成交量异常成分：相对于近期平均成交量的倍数
                volume_anomaly_component = 1.0
                if volume_zscore[i] > 1.0:  # 成交量超过1个标准差
                    volume_anomaly_component = 1.0 + min(2.0, volume_zscore[i] * 0.5)
                
                # 4. 价格加速度成分：价格上涨的加速度
                price_momentum_component = 1.0
                if price_acceleration[i] > 0:
                    price_momentum_component = 1.0 + min(1.0, price_acceleration[i] * 100)
                
                # 5. 盘口压力成分：卖一价与成交价的接近程度
                spread_component = 1.0
                if sell_price1s[i] > 0:
                    spread_ratio = (fomo_price - sell_price1s[i]) / sell_price1s[i]
                    if spread_ratio < 0.001:  # 非常接近卖一价
                        spread_component = 1.2
                    elif spread_ratio > 0.005:  # 明显高于卖一价
                        spread_component = 1.5  # 显示强烈追涨意愿
                
                # 综合FOMO分数 = 成本溢价 × 攻击性 × 成交量异常 × 价格动量 × 盘口压力
                event_fomo_score = (cost_premium_component * aggression_component * 
                                   volume_anomaly_component * price_momentum_component * 
                                   spread_component)
                
                # 加权累加
                weight = fomo_volume * (1.0 + min(1.0, volume_anomaly_component - 1.0))
                total_weighted_fomo_score += event_fomo_score * weight
                total_fomo_volume += fomo_volume
                total_fomo_amount += fomo_amount
                fomo_count += 1
    
    # 计算恐慌分数（零售在创新低时卖出）
    for i in range(num_trades):
        # 检查是否为零售交易且为卖出
        if is_retail[i] and types[i] == -1:
            # 检查是否创下新低
            if is_new_lows[i]:
                # 基础信息
                panic_volume = volumes[i]
                panic_amount = amounts[i]
                panic_price = prices[i]
                
                # 1. 成本折价成分：相对于主力买入成本的折价
                cost_discount_component = 0.0
                if not np.isnan(cost_mf_buy) and cost_mf_buy > 0:
                    cost_discount_component = (cost_mf_buy - panic_price) / atr
                
                # 2. 攻击性成分：主动卖出的惩罚
                aggression_component = 1.0
                if aggressive_sell[i]:
                    aggression_component = 1.5  # 主动卖出显示更强的恐慌
                else:
                    aggression_component = 0.7  # 被动卖出显示较弱的恐慌
                
                # 3. 成交量异常成分
                volume_anomaly_component = 1.0
                if volume_zscore[i] > 1.0:
                    volume_anomaly_component = 1.0 + min(2.0, volume_zscore[i] * 0.5)
                
                # 4. 价格加速度成分：价格下跌的加速度
                price_momentum_component = 1.0
                if price_acceleration[i] < 0:
                    price_momentum_component = 1.0 + min(1.0, abs(price_acceleration[i]) * 100)
                
                # 5. 盘口压力成分：买一价与成交价的接近程度
                spread_component = 1.0
                if buy_price1s[i] > 0:
                    spread_ratio = (buy_price1s[i] - panic_price) / buy_price1s[i]
                    if spread_ratio < 0.001:  # 非常接近买一价
                        spread_component = 1.2
                    elif spread_ratio > 0.005:  # 明显低于买一价
                        spread_component = 1.5  # 显示强烈恐慌抛售
                
                # 综合恐慌分数 = 成本折价 × 攻击性 × 成交量异常 × 价格动量 × 盘口压力
                event_panic_score = (cost_discount_component * aggression_component * 
                                    volume_anomaly_component * price_momentum_component * 
                                    spread_component)
                
                # 加权累加
                weight = panic_volume * (1.0 + min(1.0, volume_anomaly_component - 1.0))
                total_weighted_panic_score += event_panic_score * weight
                total_panic_volume += panic_volume
                total_panic_amount += panic_amount
                panic_count += 1
    
    return total_weighted_fomo_score, total_fomo_volume, total_fomo_amount, \
           total_weighted_panic_score, total_panic_volume, total_panic_amount, \
           fomo_count, panic_count

@numba.njit(cache=True)
def _numba_process_hf_clustering_and_signals(
    times: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    types: np.ndarray,
    time_gap_ns: int,
    vol_cluster_thresh: float,
    price_tolerance: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba加速版：处理高频交易的聚类（时间/成交量/价格）及对倒信号检测。
    输入为主力交易数据的NumPy数组。
    """
    n = len(times)
    trade_groups = np.zeros(n, dtype=np.int64)
    volume_groups = np.zeros(n, dtype=np.int64)
    price_groups = np.zeros(n, dtype=np.int64)
    wash_signals = np.zeros(n, dtype=np.int8)
    # 聚类初始化
    group_id = 1
    current_group_vol = 0.0
    last_time = times[0]
    trade_groups[0] = group_id
    current_group_vol = volumes[0]
    # 成交量聚类初始化
    vol_group_id = 1
    current_vol_group_vol = volumes[0]
    volume_groups[0] = vol_group_id
    # 价格聚类初始化
    price_group_id = 1
    last_price = prices[0]
    price_groups[0] = price_group_id
    # 1. 聚类循环 (从第2个元素开始)
    for i in range(1, n):
        # --- 时间聚类 ---
        if (times[i] - last_time) <= time_gap_ns and current_group_vol < vol_cluster_thresh:
            trade_groups[i] = group_id
            current_group_vol += volumes[i]
        else:
            group_id += 1
            trade_groups[i] = group_id
            current_group_vol = volumes[i]
        last_time = times[i]
        # --- 成交量聚类 (不考虑时间) ---
        if current_vol_group_vol < vol_cluster_thresh:
            volume_groups[i] = vol_group_id
            current_vol_group_vol += volumes[i]
        else:
            vol_group_id += 1
            volume_groups[i] = vol_group_id
            current_vol_group_vol = volumes[i]
        # --- 价格聚类 ---
        if abs(prices[i] - last_price) / last_price <= price_tolerance:
            price_groups[i] = price_group_id
        else:
            price_group_id += 1
            price_groups[i] = price_group_id
        last_price = prices[i]
    # 2. 对倒交易检测循环
    # 预设对倒时间阈值 1秒 (1e9 ns)
    wash_time_thresh = 1000000000
    for i in range(n - 1):
        t1, t2 = times[i], times[i+1]
        if (t2 - t1) <= wash_time_thresh:
            p1, p2 = prices[i], prices[i+1]
            v1, v2 = volumes[i], volumes[i+1]
            type1, type2 = types[i], types[i+1]
            # 对倒特征：价格接近(0.05%)，成交量相近(20%)，方向相反(1 vs -1)
            if (abs(p1 - p2) / p1 <= 0.0005):
                max_v = max(v1, v2)
                if max_v > 0 and (abs(v1 - v2) / max_v <= 0.2):
                    # 类型需一买一卖 (1:B, -1:S, 0:M)
                    if (type1 == 1 and type2 == -1) or (type1 == -1 and type2 == 1):
                        wash_signals[i] = 1
                        wash_signals[i+1] = 1
    return trade_groups, volume_groups, price_groups, wash_signals



class AdvancedFundFlowMetricsService:
    """
    【V1.4 · 主力日度净买卖股数全面捕捉版】高级资金流指标服务
    - 核心职责: 封装所有高级资金流指标的加载、计算、融合与存储逻辑。
    - 核心升级: 引入基于高频tick数据的主力日度买入股数、卖出股数及净买卖股数，
                 现在这些指标全面捕捉主力所有类型的买卖行为（包括主动攻击和被动承接/派发），
                 为累积量化提供更直接、更全面的筹码规模衡量，解决金额累积受股价影响的问题。
    - 架构优势: 实现业务逻辑与任务调度的完全解耦。
    """

    def __init__(self, debug_params: dict = None):
        self.max_lookback_days = 300
        self.debug_params = debug_params if debug_params is not None else {}

    def _get_safe_numeric_series(self, df: pd.DataFrame, col_name: str, default_value=0) -> pd.Series:
        """
        【V2.0 · 单行兼容版】类型安全的列获取辅助函数。
        修正了对单行DataFrame处理时返回标量导致后续链式调用失败的BUG。
        """
        # 彻底修正单行DataFrame问题
        if col_name not in df.columns:
            # 如果列不存在，创建一个填充了默认值的Series
            return pd.Series(default_value, index=df.index, dtype=float)
        # 使用 df[col_name] 保证返回的是Series，而不是标量，从根本上解决问题
        series = df[col_name]
        # 先转换为数值类型，再填充NaN
        return pd.to_numeric(series, errors='coerce').fillna(default_value)

    def _get_numeric_series_with_nan(self, df: pd.DataFrame, col_name: str) -> pd.Series:
        """
        安全地获取一个列作为数值型Series，并保留NaN。
        对单行DataFrame具有鲁棒性。
        """
        if col_name not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype=float)
        # 使用 df[col_name] 保证返回的是Series，而不是标量
        series = df[col_name]
        return pd.to_numeric(series, errors='coerce')

    async def run_precomputation(self, stock_code: str, is_incremental: bool, start_date_str: str = None, preloaded_minute_data: pd.DataFrame = None):
        stock_info, MetricsModel, is_incremental_final, last_metric_date, fetch_start_date = await self._initialize_context(
            stock_code, is_incremental, start_date_str
        )
        if not is_incremental_final:
            await sync_to_async(MetricsModel.objects.filter(stock=stock_info).delete)()
            DailyModel = get_daily_data_model_by_code(stock_code)
            all_dates_qs = DailyModel.objects.filter(stock=stock_info).values_list('trade_time', flat=True).order_by('trade_time')
            dates_to_process = pd.to_datetime(await sync_to_async(list)(all_dates_qs))
        else:
            mode = "部分全量" if start_date_str else "增量"
            rollback_start_date = fetch_start_date if fetch_start_date else start_date_str
            if rollback_start_date:
                await sync_to_async(MetricsModel.objects.filter(stock=stock_info, trade_time__gte=rollback_start_date).delete)()
            DailyModel = get_daily_data_model_by_code(stock_code)
            all_dates_qs = DailyModel.objects.filter(stock=stock_info, trade_time__gte=fetch_start_date).values_list('trade_time', flat=True).order_by('trade_time')
            dates_to_process = pd.to_datetime(await sync_to_async(list)(all_dates_qs))
        if dates_to_process.empty:
            return 0
        initial_history_end_date = dates_to_process.min()
        historical_metrics_df = await self._load_historical_metrics(MetricsModel, stock_info, initial_history_end_date)
        CHUNK_SIZE = 50
        # 优化点：使用列表收集 chunks，避免 O(N^2) 的 concat 操作
        metrics_chunks = []
        for i in range(0, len(dates_to_process), CHUNK_SIZE):
            chunk_dates = dates_to_process[i:i + CHUNK_SIZE]
            if chunk_dates.empty:
                continue
            chunk_start_date, chunk_end_date = chunk_dates.min(), chunk_dates.max()
            chunk_raw_data_df = await self._load_and_merge_sources(stock_info, start_date=chunk_start_date, end_date=chunk_end_date)
            if chunk_raw_data_df.empty:
                continue
            self._minute_df_daily_grouped = await self._get_daily_grouped_minute_data(stock_info, chunk_raw_data_df.index)
            chunk_new_metrics_df, _, _ = self._synthesize_and_forge_metrics(stock_code, chunk_raw_data_df)
            if not chunk_new_metrics_df.empty:
                metrics_chunks.append(chunk_new_metrics_df)
        if hasattr(self, '_minute_df_daily_grouped'):
            del self._minute_df_daily_grouped
        if not metrics_chunks:
            return 0
        # 一次性合并所有结果
        all_new_core_metrics_df = pd.concat(metrics_chunks)
        full_sequence_for_derivatives = pd.concat([historical_metrics_df, all_new_core_metrics_df])
        full_sequence_for_derivatives.sort_index(inplace=True)
        final_metrics_df = self._calculate_derivatives(stock_code, full_sequence_for_derivatives)
        chunk_to_save = final_metrics_df[final_metrics_df.index.isin(all_new_core_metrics_df.index)]
        total_processed_count = await self._prepare_and_save_data(stock_info, MetricsModel, chunk_to_save)
        return total_processed_count

    async def _initialize_context(self, stock_code: str, is_incremental: bool, start_date_str: str = None):
        from datetime import datetime
        stock_info = await sync_to_async(StockInfo.objects.get)(stock_code=stock_code)
        MetricsModel = get_advanced_fund_flow_metrics_model_by_code(stock_code)
        last_metric_date = None
        fetch_start_date = None
        # 移除所有调试性质的print语句
        if start_date_str:
            try:
                start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                is_incremental = True
                last_metric_date = start_date_obj - timedelta(days=1)
                fetch_start_date = start_date_obj - timedelta(days=self.max_lookback_days)
            except (ValueError, TypeError):
                is_incremental = True
        if is_incremental and not start_date_str:
            @sync_to_async(thread_sensitive=True)
            def get_latest_metric_async(model, stock_info_obj):
                try:
                    return model.objects.filter(stock=stock_info_obj).latest('trade_time')
                except model.DoesNotExist:
                    return None
            latest_metric = await get_latest_metric_async(MetricsModel, stock_info)
            if latest_metric:
                last_metric_date = latest_metric.trade_time
                fetch_start_date = last_metric_date - timedelta(days=self.max_lookback_days)
            else:
                is_incremental = False
                fetch_start_date = None
        return stock_info, MetricsModel, is_incremental, last_metric_date, fetch_start_date

    async def _load_and_merge_sources(self, stock_info, data_dfs: dict, base_daily_df: pd.DataFrame):
        """
        【V2.4 · 净流量悖论修复版】
        - 核心修复: 解决了“净流量悖论”。在 `standardize_and_prepare` 中，彻底移除了为 THS 和 DC 数据源
                     从“净额”数据反推“买入/卖出毛坯额”的错误逻辑。
        - 核心思想: 停止凭空捏造数据。系统现在只使用 Tushare 提供的真实“毛坯”数据进行需要 gross flow 的计算。
                     如果 Tushare 数据缺失，相关指标将正确地输出为空(NaN)，而不是基于虚假数据得出错误结论，
                     从根本上保证了下游概率成本等核心指标的数据纯净性。
        """
        def standardize_and_prepare(df: pd.DataFrame, source: str) -> pd.DataFrame:
            if df.empty: return df
            df['trade_time'] = pd.to_datetime(df['trade_time'])
            cols_to_numeric = [col for col in df.columns if col != 'trade_time' and 'code' not in col and 'name' not in col]
            df[cols_to_numeric] = df[cols_to_numeric].apply(pd.to_numeric, errors='coerce')
            required_amount_cols = [
                'buy_sm_amount', 'sell_sm_amount', 'buy_md_amount', 'sell_md_amount',
                'buy_lg_amount', 'sell_lg_amount', 'buy_elg_amount', 'sell_elg_amount',
            ]
            required_net_amount_cols = [
                'net_mf_amount', 'net_amount', 'net_amount_main', 'net_amount_xl',
                'net_amount_lg', 'net_amount_md', 'net_amount_sm', 'trade_count'
            ]
            for col in required_amount_cols + required_net_amount_cols:
                if col not in df.columns:
                    df[col] = 0.0
            if source == 'tushare':
                df['net_flow_tushare'] = df['net_mf_amount']
                df['main_force_net_flow_tushare'] = (df['buy_lg_amount'] - df['sell_lg_amount']) + (df['buy_elg_amount'] - df['sell_elg_amount'])
                df['retail_net_flow_tushare'] = (df['buy_sm_amount'] - df['sell_sm_amount']) + (df['buy_md_amount'] - df['sell_md_amount'])
                df['net_xl_amount_tushare'] = df['buy_elg_amount'] - df['sell_elg_amount']
                df['net_lg_amount_tushare'] = df['buy_lg_amount'] - df['sell_lg_amount']
                df['net_md_amount_tushare'] = df['buy_md_amount'] - df['sell_md_amount']
                df['net_sh_amount_tushare'] = df['buy_sm_amount'] - df['sell_sm_amount']
                return df
            elif source == 'ths':
                df['net_flow_ths'] = df['net_amount']
                df['main_force_net_flow_ths'] = df['buy_lg_amount']
                df['retail_net_flow_ths'] = df['buy_md_amount'] + df['buy_sm_amount']
                # 移除所有从净额反推毛坯额的错误逻辑
                # 不再捏造 buy/sell amount，只保留真实的 net amount
                df['net_lg_amount_ths'] = df['buy_lg_amount']
                df['net_md_amount_ths'] = df['buy_md_amount']
                df['net_sh_amount_ths'] = df['buy_sm_amount']
                # 确保 buy/sell amount 列存在但为空，以维持 schema 一致性，但不填充虚假数据
                for col in ['buy_lg_amount', 'sell_lg_amount', 'buy_md_amount', 'sell_md_amount', 'buy_sm_amount', 'sell_sm_amount']:
                    if col not in df.columns: df[col] = np.nan
                return df
            elif source == 'dc':
                df['main_force_net_flow_dc'] = df['net_amount']
                df['retail_net_flow_dc'] = df['net_amount_md'] + df['net_amount_sm']
                df['net_flow_dc'] = df['main_force_net_flow_dc'] + df['retail_net_flow_dc']
                # 移除所有从净额反推毛坯额的错误逻辑
                # 不再捏造 buy/sell amount，只保留真实的 net amount
                df['net_xl_amount_dc'] = df['net_amount_xl']
                df['net_lg_amount_dc'] = df['net_amount_lg']
                df['net_md_amount_dc'] = df['net_amount_md']
                df['net_sh_amount_dc'] = df['net_amount_sm']
                # 确保 buy/sell amount 列存在但为空
                for col in ['buy_elg_amount', 'sell_elg_amount', 'buy_lg_amount', 'sell_lg_amount', 'buy_md_amount', 'sell_md_amount', 'buy_sm_amount', 'sell_sm_amount']:
                    if col not in df.columns: df[col] = np.nan
                return df
            return df
        df_tushare = standardize_and_prepare(data_dfs['tushare'], 'tushare')
        df_ths = standardize_and_prepare(data_dfs['ths'], 'ths')
        df_dc = standardize_and_prepare(data_dfs['dc'], 'dc')
        if df_tushare.empty:
            return pd.DataFrame()
        merged_df = df_tushare
        other_flow_dfs = [df for df in [df_ths, df_dc] if not df.empty]
        if other_flow_dfs:
            for right_df in other_flow_dfs:
                overlap_cols = merged_df.columns.intersection(right_df.columns).drop('trade_time', errors='ignore')
                right_df_cleaned = right_df.drop(columns=overlap_cols, errors='ignore')
                merged_df = pd.merge(merged_df, right_df_cleaned, on='trade_time', how='left')
        merged_df = merged_df.sort_values('trade_time').set_index('trade_time')
        if not base_daily_df.empty:
            base_daily_df_copy = base_daily_df.copy()
            merged_df.index = pd.to_datetime(merged_df.index).normalize()
            base_daily_df_copy.index = pd.to_datetime(base_daily_df_copy.index).normalize()
            overlap_cols = merged_df.columns.intersection(base_daily_df_copy.columns)
            merged_df = merged_df.join(base_daily_df_copy.drop(columns=overlap_cols, errors='ignore'), how='left')
        return merged_df

    def _prepare_behavioral_data(self, intraday_data: pd.DataFrame, daily_data: pd.Series, tick_data: pd.DataFrame = None, level5_data: pd.DataFrame = None, realtime_data: pd.DataFrame = None) -> tuple:
        """
        【V64.0 · 特征工程一体化】
        - 核心重构: 净化此方法的职责。移除所有特征衍生计算（如OFI, imbalance等），
                     使其回归到只负责合并原始高频数据源的单一职责，为下游统一的特征工程中心提供纯净的输入。
        """
        import numpy as np
        daily_total_volume = daily_data.get('vol', 0) * 100
        daily_total_amount = pd.to_numeric(daily_data.get('amount', 0), errors='coerce') * 1000
        daily_vwap = daily_total_amount / daily_total_volume if daily_total_volume > 0 else np.nan
        atr = daily_data.get('atr_14d')
        day_open, day_close = daily_data.get('open_qfq'), daily_data.get('close_qfq')
        day_high, day_low = daily_data.get('high_qfq'), daily_data.get('low_qfq')
        raw_hf_df = pd.DataFrame()
        if tick_data is not None and not tick_data.empty and level5_data is not None and not level5_data.empty:
            merged_hf = pd.merge_asof(
                tick_data.sort_index(), level5_data.sort_index(),
                left_index=True, right_index=True, direction='backward'
            ).dropna(subset=['buy_price1', 'sell_price1', 'amount', 'volume'])
            if realtime_data is not None and not realtime_data.empty and not merged_hf.empty:
                realtime_prepped = realtime_data[['volume']].copy()
                realtime_prepped['snapshot_time'] = realtime_prepped.index
                merged_hf = pd.merge_asof(
                    merged_hf, realtime_prepped, left_index=True, right_index=True,
                    direction='backward', suffixes=('_tick', '_realtime')
                )
            if not merged_hf.empty:
                merged_hf.rename(columns={'volume_tick': 'volume'}, inplace=True)
                raw_hf_df = merged_hf
        common_data = {
            'daily_total_volume': daily_total_volume, 'daily_total_amount': daily_total_amount,
            'daily_vwap': daily_vwap, 'atr': atr, 'day_open': day_open, 'day_close': day_close,
            'day_high': day_high, 'day_low': day_low
        }
        return raw_hf_df, common_data

    def _engineer_hf_features(self, raw_hf_df: pd.DataFrame, daily_total_volume: float, context: dict = None) -> tuple[pd.DataFrame, dict]:
        features = {
            'mf_trades': pd.DataFrame(), 'buy_trades_mask': pd.Series(dtype=bool),
            'sell_trades_mask': pd.Series(dtype=bool), 'total_mf_vol': 0.0,
            'mf_buy_vol': 0.0, 'mf_sell_vol': 0.0, 'offensive_volume': 0.0,
            'passive_volume': 0.0, 'hf_mf_buy_vwap': np.nan, 'hf_mf_sell_vwap': np.nan,
            'main_force_aggressive_buy_volume': 0.0,
            'main_force_aggressive_sell_volume': 0.0,
            'main_force_passive_buy_volume': 0.0,
            'main_force_passive_sell_volume': 0.0,
            'main_force_avg_price_impact': np.nan,
            'main_force_daily_buy_amount': 0.0,
            'main_force_daily_sell_amount': 0.0,
            'main_force_daily_buy_volume': 0.0,
            'main_force_daily_sell_volume': 0.0,
            'main_force_net_flow_ratio': 0.0,
            'main_force_camouflage_ratio': 0.0,
            'main_force_clustered_buy_volume': 0.0,
            'main_force_clustered_sell_volume': 0.0,
        }
        if raw_hf_df is None or raw_hf_df.empty:
            return pd.DataFrame(), features
        hf_analysis_df = raw_hf_df.copy()
        hf_analysis_df['mid_price'] = (hf_analysis_df['buy_price1'] + hf_analysis_df['sell_price1']) / 2
        hf_analysis_df['prev_mid_price'] = hf_analysis_df['mid_price'].shift(1)
        hf_analysis_df['mid_price_change'] = hf_analysis_df['mid_price'].diff()
        buy_pressure_quote = np.where(hf_analysis_df['mid_price'] >= hf_analysis_df['prev_mid_price'], hf_analysis_df['buy_volume1'].shift(1), 0)
        sell_pressure_quote = np.where(hf_analysis_df['mid_price'] <= hf_analysis_df['prev_mid_price'], hf_analysis_df['sell_volume1'].shift(1), 0)
        hf_analysis_df['ofi'] = buy_pressure_quote - sell_pressure_quote
        # 重新计算平均交易金额和交易量，供本方法内部使用
        avg_trade_amount = hf_analysis_df['amount'].mean() if not hf_analysis_df.empty else 0.0
        avg_trade_volume = hf_analysis_df['volume'].mean()
        # 调用新方法识别主力交易和散户交易，并传入 context
        is_main_force_trade, is_retail_trade = AdvancedFundFlowMetricsService._identify_trade_participants(hf_analysis_df, context)
        net_active_volume_series = pd.Series(0.0, index=hf_analysis_df.index)
        active_buy_mask = hf_analysis_df['price'] >= hf_analysis_df['sell_price1']
        active_sell_mask = hf_analysis_df['price'] <= hf_analysis_df['buy_price1']
        net_active_volume_series.loc[active_buy_mask] = hf_analysis_df.loc[active_buy_mask, 'volume']
        net_active_volume_series.loc[active_sell_mask] = -hf_analysis_df.loc[active_sell_mask, 'volume']
        hf_analysis_df['net_active_volume'] = net_active_volume_series
        hf_analysis_df['main_force_ofi'] = np.where(is_main_force_trade, hf_analysis_df['net_active_volume'], 0)
        hf_analysis_df['retail_ofi'] = np.where(is_retail_trade, hf_analysis_df['net_active_volume'], 0)
        hf_analysis_df['prev_a1_p'] = hf_analysis_df['sell_price1'].shift(1)
        hf_analysis_df['prev_b1_p'] = hf_analysis_df['buy_price1'].shift(1)
        hf_analysis_df['prev_a1_v'] = hf_analysis_df['sell_volume1'].shift(1)
        hf_analysis_df['prev_b1_v'] = hf_analysis_df['buy_volume1'].shift(1)
        hf_analysis_df['imbalance'] = np.nan
        hf_analysis_df['liquidity_supply_ratio'] = np.nan
        try:
            required_level5_cols = [f'{side}_{col}{i}' for i in range(1, 6) for side in ['buy', 'sell'] for col in ['volume', 'price']]
            if all(col in hf_analysis_df.columns for col in required_level5_cols):
                weighted_buy_vol = pd.Series(0, index=hf_analysis_df.index)
                weighted_sell_vol = pd.Series(0, index=hf_analysis_df.index)
                total_buy_value = pd.Series(0, index=hf_analysis_df.index)
                total_sell_value = pd.Series(0, index=hf_analysis_df.index)
                for i in range(1, 6):
                    weight = 1 / i
                    weighted_buy_vol += hf_analysis_df[f'buy_volume{i}'] * weight
                    weighted_sell_vol += hf_analysis_df[f'sell_volume{i}'] * weight
                    total_buy_value += hf_analysis_df[f'buy_volume{i}'] * hf_analysis_df[f'buy_price{i}']
                    total_sell_value += hf_analysis_df[f'buy_volume{i}'] * hf_analysis_df[f'sell_price{i}']
                sum_weighted_vol = (weighted_buy_vol + weighted_sell_vol)
                if (sum_weighted_vol > 1e-9).any():
                    hf_analysis_df['imbalance'] = (weighted_buy_vol - weighted_sell_vol) / sum_weighted_vol.replace(0, np.nan)
                if (total_sell_value > 1e-9).any():
                    hf_analysis_df['liquidity_supply_ratio'] = total_buy_value / total_sell_value.replace(0, np.nan)
        except Exception as e:
            pass
        mf_trades = hf_analysis_df[is_main_force_trade].copy()
        if mf_trades.empty:
            return hf_analysis_df, features
        features['mf_trades'] = mf_trades
        # --- 极致精细的主力意图识别逻辑 ---
        # 1. 订单再生模式检测：主力在撤单后立即重新挂单进行反复吸筹/出货
        hf_analysis_df['order_renewal_signal'] = 0
        if len(hf_analysis_df) > 1:
            # 检测主动买入后的卖盘委托变化
            buy_after_ask_change = ((hf_analysis_df['type'] == 'B') & 
                                   (hf_analysis_df['sell_volume1'] < hf_analysis_df['sell_volume1'].shift(1).where(hf_analysis_df.index > hf_analysis_df.index[0], 0)) &
                                   (hf_analysis_df['sell_volume1'].shift(-1) > hf_analysis_df['sell_volume1'] * 1.2)).shift(1, fill_value=False)
            # 检测主动卖出后的买盘委托变化
            sell_after_bid_change = ((hf_analysis_df['type'] == 'S') & 
                                    (hf_analysis_df['buy_volume1'] < hf_analysis_df['buy_volume1'].shift(1).where(hf_analysis_df.index > hf_analysis_df.index[0], 0)) &
                                    (hf_analysis_df['buy_volume1'].shift(-1) > hf_analysis_df['buy_volume1'] * 1.2)).shift(1, fill_value=False)
            hf_analysis_df.loc[buy_after_ask_change, 'order_renewal_signal'] = 1  # 买入后卖盘再生
            hf_analysis_df.loc[sell_after_bid_change, 'order_renewal_signal'] = -1  # 卖出后买盘再生
        # 2. 多维度拆单集群检测
        hf_analysis_df['trade_group'] = 0
        hf_analysis_df['volume_group'] = 0
        hf_analysis_df['price_group'] = 0
        group_id = 1
        volume_group_id = 1
        price_group_id = 1
        time_gap_threshold = pd.Timedelta('2s')
        # 定义用于聚类分析的绝对主力金额阈值
        ABSOLUTE_MAIN_FORCE_AMOUNT_FOR_CLUSTERING = 200000
        volume_cluster_threshold = 0.7 * ABSOLUTE_MAIN_FORCE_AMOUNT_FOR_CLUSTERING / mf_trades['price'].mean() if not mf_trades.empty else 10000
        price_tolerance = 0.001  # 0.1%价格容忍度
        last_time = None
        last_price = None
        current_group_volume = 0
        current_volume_group_volume = 0
        for idx, row in mf_trades.iterrows():
            # 时间聚类
            if last_time is not None and (idx - last_time) <= time_gap_threshold and current_group_volume < volume_cluster_threshold:
                hf_analysis_df.at[idx, 'trade_group'] = group_id
                current_group_volume += row['volume']
            else:
                group_id += 1
                hf_analysis_df.at[idx, 'trade_group'] = group_id
                current_group_volume = row['volume']
            last_time = idx
            # 成交量聚类（不考虑时间）
            if current_volume_group_volume < volume_cluster_threshold:
                hf_analysis_df.at[idx, 'volume_group'] = volume_group_id
                current_volume_group_volume += row['volume']
            else:
                volume_group_id += 1
                hf_analysis_df.at[idx, 'volume_group'] = volume_group_id
                current_volume_group_volume = row['volume']
            # 价格聚类
            if last_price is not None and abs(row['price'] - last_price) / last_price <= price_tolerance:
                hf_analysis_df.at[idx, 'price_group'] = price_group_id
            else:
                price_group_id += 1
                hf_analysis_df.at[idx, 'price_group'] = price_group_id
            last_price = row['price']
        # 3. 对倒交易检测（自买自卖制造活跃假象）
        hf_analysis_df['wash_trade_signal'] = 0
        if len(mf_trades) > 1:
            mf_indices = mf_trades.index
            for i in range(len(mf_indices) - 1):
                idx1 = mf_indices[i]
                idx2 = mf_indices[i + 1]
                if (idx2 - idx1) <= pd.Timedelta('1s'):
                    row1 = mf_trades.loc[idx1]
                    row2 = mf_trades.loc[idx2]
                    # 对倒特征：时间接近、价格相同或接近、成交量相近、方向相反
                    if (abs(row1['price'] - row2['price']) / row1['price'] <= 0.0005 and
                        abs(row1['volume'] - row2['volume']) / max(row1['volume'], row2['volume']) <= 0.2 and
                        ((row1['type'] == 'B' and row2['type'] == 'S') or (row1['type'] == 'S' and row2['type'] == 'B'))):
                        hf_analysis_df.loc[idx1, 'wash_trade_signal'] = 1
                        hf_analysis_df.loc[idx2, 'wash_trade_signal'] = 1
        # 4. 复杂的中性单意图识别
        neutral_mask = mf_trades['type'] == 'M'
        # 获取与mf_trades索引对齐的订单再生信号和对倒信号
        mf_order_renewal_signal = hf_analysis_df.loc[mf_trades.index, 'order_renewal_signal']
        mf_wash_trade_signal = hf_analysis_df.loc[mf_trades.index, 'wash_trade_signal']
        # 初始化这些DataFrame，以防后续concat操作时它们为空
        mf_aggressive_buy_trades = pd.DataFrame()
        mf_aggressive_sell_trades = pd.DataFrame()
        mf_passive_buy_trades = pd.DataFrame()
        mf_passive_sell_trades = pd.DataFrame()
        # 4.1 价格穿透深度分析
        price_penetration_buy = neutral_mask & (
            (mf_trades['price'] >= mf_trades['prev_a1_p'] * 0.999) &  # 轻微穿透卖一
            (mf_trades['price'] <= mf_trades['sell_price1'] * 1.001)  # 在当前卖一附近
        )
        price_penetration_sell = neutral_mask & (
            (mf_trades['price'] <= mf_trades['prev_b1_p'] * 1.001) &  # 轻微穿透买一
            (mf_trades['price'] >= mf_trades['buy_price1'] * 0.999)   # 在当前买一附近
        )
        # 4.2 盘口压力分析
        bid_ask_pressure_buy = neutral_mask & (
            (mf_trades['buy_volume1'] > mf_trades['sell_volume1'] * 1.5) &  # 买盘压力大
            (mf_trades['mid_price_change'] >= 0)  # 价格在上涨
        )
        bid_ask_pressure_sell = neutral_mask & (
            (mf_trades['sell_volume1'] > mf_trades['buy_volume1'] * 1.5) &  # 卖盘压力大
            (mf_trades['mid_price_change'] < 0)  # 价格在下跌
        )
        # 4.3 结合订单再生信号
        order_renewal_buy = neutral_mask & (mf_order_renewal_signal == 1)
        order_renewal_sell = neutral_mask & (mf_order_renewal_signal == -1)
        # 综合判断被动买卖
        passive_buy_from_neutral = (price_penetration_buy | bid_ask_pressure_buy | order_renewal_buy) & \
                                   (mf_trades['volume'] > 0.3 * avg_trade_volume) & \
                                   (mf_wash_trade_signal == 0)
        passive_sell_from_neutral = (price_penetration_sell | bid_ask_pressure_sell | order_renewal_sell) & \
                                    (mf_trades['volume'] > 0.3 * avg_trade_volume) & \
                                    (mf_wash_trade_signal == 0)
        # 剩余中性单按价格变化方向划分（保守策略）
        remaining_neutral = mf_trades[neutral_mask & ~passive_buy_from_neutral & ~passive_sell_from_neutral &
                                     (mf_wash_trade_signal == 0)]
        remaining_passive_buy = remaining_neutral[remaining_neutral['mid_price_change'] >= 0]
        remaining_passive_sell = remaining_neutral[remaining_neutral['mid_price_change'] < 0]
        mf_passive_buy_trades = pd.concat([mf_passive_buy_trades, remaining_passive_buy])
        mf_passive_sell_trades = pd.concat([mf_passive_sell_trades, remaining_passive_sell])
        # 5. 多层级集群分析
        # 确保 group_stats 和 volume_group_stats 在任何情况下都已定义
        group_stats = pd.DataFrame()
        volume_group_stats = pd.DataFrame()
        if not mf_trades.empty:
            group_stats = hf_analysis_df[hf_analysis_df['trade_group'] > 0].groupby('trade_group').agg({
                'volume': 'sum',
                'amount': 'sum',
                'type': lambda x: x.mode()[0] if not x.mode().empty else 'M',
                'net_active_volume': 'sum',
                'order_renewal_signal': 'mean',
                'wash_trade_signal': 'max'
            })
            volume_group_stats = hf_analysis_df[hf_analysis_df['volume_group'] > 0].groupby('volume_group').agg({
                'volume': 'sum',
                'amount': 'sum',
                'type': lambda x: x.mode()[0] if not x.mode().empty else 'M',
                'net_active_volume': 'sum'
            })
        # 6. 精准的主力交易分类（排除对倒，考虑订单再生）
        # 主动买入：排除对倒，考虑订单再生
        mf_aggressive_buy_trades = mf_trades[(mf_trades['type'] == 'B') & 
                                            (mf_wash_trade_signal == 0)]
        # 主动卖出：排除对倒，考虑订单再生
        mf_aggressive_sell_trades = mf_trades[(mf_trades['type'] == 'S') & 
                                             (mf_wash_trade_signal == 0)]
        # 被动买入：综合判断
        mf_passive_buy_trades = mf_trades[passive_buy_from_neutral & 
                                         (mf_wash_trade_signal == 0)]
        # 被动卖出：综合判断
        mf_passive_sell_trades = mf_trades[passive_sell_from_neutral & 
                                          (mf_wash_trade_signal == 0)]
        # 剩余中性单按价格变化方向划分（保守策略）
        remaining_neutral = mf_trades[neutral_mask & ~passive_buy_from_neutral & ~passive_sell_from_neutral &
                                     (mf_wash_trade_signal == 0)]
        remaining_passive_buy = remaining_neutral[remaining_neutral['mid_price_change'] >= 0]
        remaining_passive_sell = remaining_neutral[remaining_neutral['mid_price_change'] < 0]
        mf_passive_buy_trades = pd.concat([mf_passive_buy_trades, remaining_passive_buy])
        mf_passive_sell_trades = pd.concat([mf_passive_sell_trades, remaining_passive_sell])
        # 7. 极致精细的主力日度数据计算
        # 7.1 时间聚类集群成交量
        # 探针：检查 group_stats 在使用前是否为空
        should_probe = context['debug'].get('should_probe', False)
        if should_probe:
            stock_code = context['debug']['stock_code']
            current_date = context['daily_data'].name.date()
            print(f"\n--- [探针 _engineer_hf_features - Before clustered_buy_groups] {stock_code} {current_date} ---")
            print(f"  - mf_trades.empty: {mf_trades.empty}")
            print(f"  - group_stats.empty: {group_stats.empty}")
            print(f"  - group_stats columns: {group_stats.columns.tolist()}")
            if not group_stats.empty:
                print(f"  - group_stats head:\n{group_stats.head().to_string()}")
            else:
                print(f"  - group_stats is empty.")
            print(f"--- [探针 _engineer_hf_features - End Before clustered_buy_groups] ---")

        clustered_buy_groups = group_stats[(group_stats['net_active_volume'] > 0) & 
                                          (group_stats['volume'] >= volume_cluster_threshold) &
                                          (group_stats['wash_trade_signal'] == 0)]
        clustered_sell_groups = group_stats[(group_stats['net_active_volume'] < 0) & 
                                           (group_stats['volume'] >= volume_cluster_threshold) &
                                           (group_stats['wash_trade_signal'] == 0)]
        # 7.2 成交量聚类集群（不考虑时间）
        volume_clustered_buy = volume_group_stats[(volume_group_stats['net_active_volume'] > 0) & 
                                                 (volume_group_stats['volume'] >= volume_cluster_threshold)]
        volume_clustered_sell = volume_group_stats[(volume_group_stats['net_active_volume'] < 0) & 
                                                  (volume_group_stats['volume'] >= volume_cluster_threshold)]
        # 7.3 计算集群成交量（取两种聚类方法的并集）
        all_clustered_buy_volume = max(
            clustered_buy_groups['volume'].sum() if not clustered_buy_groups.empty else 0,
            volume_clustered_buy['volume'].sum() if not volume_clustered_buy.empty else 0
        )
        all_clustered_sell_volume = max(
            clustered_sell_groups['volume'].sum() if not clustered_sell_groups.empty else 0,
            volume_clustered_sell['volume'].sum() if not volume_clustered_sell.empty else 0
        )
        features['main_force_clustered_buy_volume'] = all_clustered_buy_volume
        features['main_force_clustered_sell_volume'] = all_clustered_sell_volume
        # 7.4 主力日度数据计算（优先级：集群识别 > 订单再生增强 > 基础分类）
        # 买入端计算
        if all_clustered_buy_volume > 0:
            # 使用集群识别结果
            clustered_buy_amount = clustered_buy_groups['amount'].sum() if not clustered_buy_groups.empty else 0
            volume_clustered_buy_amount = volume_clustered_buy['amount'].sum() if not volume_clustered_buy.empty else 0
            features['main_force_daily_buy_amount'] = max(clustered_buy_amount, volume_clustered_buy_amount)
            features['main_force_daily_buy_volume'] = all_clustered_buy_volume
        else:
            # 基础分类计算，但考虑订单再生增强
            base_buy_amount = mf_aggressive_buy_trades['amount'].sum() + mf_passive_buy_trades['amount'].sum()
            base_buy_volume = mf_aggressive_buy_trades['volume'].sum() + mf_passive_buy_trades['volume'].sum()
            # 查找订单再生相关的交易
            order_renewal_buy_trades = mf_trades[mf_order_renewal_signal == 1]
            renewal_buy_amount = order_renewal_buy_trades['amount'].sum()
            renewal_buy_volume = order_renewal_buy_trades['volume'].sum()
            # 如果订单再生交易量显著，则增强买入信号
            if renewal_buy_volume > 0.3 * base_buy_volume:
                features['main_force_daily_buy_amount'] = base_buy_amount + renewal_buy_amount * 0.7  # 打折计入
                features['main_force_daily_buy_volume'] = base_buy_volume + renewal_buy_volume * 0.7
            else:
                features['main_force_daily_buy_amount'] = base_buy_amount
                features['main_force_daily_buy_volume'] = base_buy_volume
        # 卖出端计算
        if all_clustered_sell_volume > 0:
            clustered_sell_amount = clustered_sell_groups['amount'].sum() if not clustered_sell_groups.empty else 0
            volume_clustered_sell_amount = volume_clustered_sell['amount'].sum() if not volume_clustered_sell.empty else 0
            features['main_force_daily_sell_amount'] = max(clustered_sell_amount, volume_clustered_sell_amount)
            features['main_force_daily_sell_volume'] = all_clustered_sell_volume
        else:
            base_sell_amount = mf_aggressive_sell_trades['amount'].sum() + mf_passive_sell_trades['amount'].sum()
            base_sell_volume = mf_aggressive_sell_trades['volume'].sum() + mf_passive_sell_trades['volume'].sum()
            order_renewal_sell_trades = mf_trades[mf_order_renewal_signal == -1]
            renewal_sell_amount = order_renewal_sell_trades['amount'].sum()
            renewal_sell_volume = order_renewal_sell_trades['volume'].sum()
            if renewal_sell_volume > 0.3 * base_sell_volume:
                features['main_force_daily_sell_amount'] = base_sell_amount + renewal_sell_amount * 0.7
                features['main_force_daily_sell_volume'] = base_sell_volume + renewal_sell_volume * 0.7
            else:
                features['main_force_daily_sell_amount'] = base_sell_amount
                features['main_force_daily_sell_volume'] = base_sell_volume
        # 8. 计算伪装比例和净流入比率
        total_camouflage_volume = mf_trades[(mf_trades['volume'] < 0.5 * avg_trade_volume) & 
                                           (mf_wash_trade_signal == 0)]['volume'].sum()
        # 修正 total_mf_volume 的计算，直接使用 mf_trades 中非对倒交易的成交量
        total_mf_volume_for_camouflage = mf_trades[mf_wash_trade_signal == 0]['volume'].sum()
        features['main_force_camouflage_ratio'] = total_camouflage_volume / total_mf_volume_for_camouflage if total_mf_volume_for_camouflage > 0 else 0
        total_mf_amount = features['main_force_daily_buy_amount'] + features['main_force_daily_sell_amount']
        features['main_force_net_flow_ratio'] = (features['main_force_daily_buy_amount'] - features['main_force_daily_sell_amount']) / total_mf_amount if total_mf_amount > 0 else 0
        # 保留原有特征计算
        features['main_force_aggressive_buy_volume'] = mf_aggressive_buy_trades['volume'].sum()
        features['main_force_aggressive_sell_volume'] = mf_aggressive_sell_trades['volume'].sum()
        features['main_force_passive_buy_volume'] = mf_passive_buy_trades['volume'].sum()
        features['main_force_passive_sell_volume'] = mf_passive_sell_trades['volume'].sum()
        features['offensive_volume'] = features['main_force_aggressive_buy_volume'] + features['main_force_aggressive_sell_volume']
        features['passive_volume'] = features['main_force_passive_buy_volume'] + features['main_force_passive_sell_volume']
        total_mf_buy_trades_for_vwap = pd.concat([mf_aggressive_buy_trades, mf_passive_buy_trades])
        if not total_mf_buy_trades_for_vwap.empty and total_mf_buy_trades_for_vwap['volume'].sum() > 0:
            features['hf_mf_buy_vwap'] = (total_mf_buy_trades_for_vwap['price'] * total_mf_buy_trades_for_vwap['volume']).sum() / total_mf_buy_trades_for_vwap['volume'].sum()
        total_mf_sell_trades_for_vwap = pd.concat([mf_aggressive_sell_trades, mf_passive_sell_trades])
        if not total_mf_sell_trades_for_vwap.empty and total_mf_sell_trades_for_vwap['volume'].sum() > 0:
            features['hf_mf_sell_vwap'] = (total_mf_sell_trades_for_vwap['price'] * total_mf_sell_trades_for_vwap['volume']).sum() / total_mf_sell_trades_for_vwap['volume'].sum()
        features['total_mf_vol'] = features['main_force_daily_buy_volume'] + features['main_force_daily_sell_volume']
        mf_trades['price_impact'] = np.nan
        offensive_buy_mask = mf_aggressive_buy_trades.index
        offensive_sell_mask = mf_aggressive_sell_trades.index
        mf_trades.loc[offensive_buy_mask, 'price_impact'] = (mf_trades.loc[offensive_buy_mask, 'price'] - mf_trades.loc[offensive_buy_mask, 'prev_mid_price']).values
        mf_trades.loc[offensive_sell_mask, 'price_impact'] = (mf_trades.loc[offensive_sell_mask, 'prev_mid_price'] - mf_trades.loc[offensive_sell_mask, 'price']).values
        mf_trades['price_impact'] = mf_trades['price_impact'].clip(lower=0)
        if features['offensive_volume'] > 0:
            aggressive_trades_for_impact = mf_trades[mf_trades['type'].isin(['B', 'S'])]
            if not aggressive_trades_for_impact.empty and aggressive_trades_for_impact['volume'].sum() > 0:
                features['main_force_avg_price_impact'] = (aggressive_trades_for_impact['price_impact'] * aggressive_trades_for_impact['volume']).sum() / aggressive_trades_for_impact['volume'].sum()
        return hf_analysis_df, features

    async def _get_daily_grouped_minute_data(self, stock_info: StockInfo, date_index: pd.DatetimeIndex, fetch_full_cols: bool = True, tick_data_map: dict = None, level5_data_map: dict = None, minute_data_map: dict = None):
        """
        【V1.14 · 日内数据回退增强版】不再查询数据库，仅处理由上游任务传入的日内数据maps。
        - 核心重构: 移除所有数据库查询逻辑，职责单一化为数据处理与聚合。
        - 核心逻辑: 遍历所需日期，优先尝试逐笔数据，若处理失败则回退到分钟数据。
        - 核心修复: 修正逐笔数据与Level5数据合并后，价格、成交量、成交额列名未被 `suffixes` 参数重命名的问题。
                    这些列名应保持原始名称，避免 `KeyError`。
        - 核心增强: 引入逐笔数据处理失败回退机制，确保分钟数据在逐笔数据不可用时能被利用。
        """
        import pandas as pd
        from django.utils import timezone
        if date_index.empty:
            return {}
        intraday_data_map = {}
        for date_obj in date_index.date:
            processed_with_tick_data = False
            if tick_data_map and date_obj in tick_data_map:
                try:
                    tick_df = tick_data_map[date_obj].copy()
                    if not all(col in tick_df.columns for col in ['price', 'volume', 'amount']):
                        logger.warning(f"[{stock_info.stock_code}] [资金流服务] 日期 {date_obj} 逐笔数据缺少'price', 'volume'或'amount'列，将尝试回退到分钟数据。")
                        raise ValueError("Missing essential tick data columns")
                    current_price_col = 'price'
                    current_volume_col = 'volume'
                    current_amount_col = 'amount'
                    has_original_type = 'type' in tick_df.columns
                    if level5_data_map and date_obj in level5_data_map:
                        level5_df = level5_data_map[date_obj]
                        tick_df_sorted = tick_df.sort_index()
                        level5_df_sorted = level5_df.sort_index()
                        merged_df_temp = pd.merge_asof(
                            tick_df_sorted.reset_index(),
                            level5_df_sorted.reset_index(),
                            on='trade_time',
                            direction='backward',
                            suffixes=('_tick', '_level5')
                        )
                        tick_df = merged_df_temp.set_index('trade_time')
                        if 'sell_price1' in tick_df.columns and 'buy_price1' in tick_df.columns:
                            conditions = [tick_df[current_price_col] >= tick_df['sell_price1'], tick_df[current_price_col] <= tick_df['buy_price1']]
                            choices = ['B', 'S']
                            tick_df['type'] = np.select(conditions, choices, default='M')
                        else:
                            logger.warning(f"[{stock_info.stock_code}] [资金流服务] 日期 {date_obj} 合并Level5数据后缺少买卖价格，无法重新评估'type'。")
                            if not has_original_type:
                                tick_df['type'] = 'M'
                    else:
                        if not has_original_type:
                            logger.warning(f"[{stock_info.stock_code}] [资金流服务] 日期 {date_obj} 缺少Level5数据且原始逐笔数据无'type'列，'type'将默认为'M'。")
                            tick_df['type'] = 'M'
                    if 'type' not in tick_df.columns:
                        logger.warning(f"[{stock_info.stock_code}] [资金流服务] 日期 {date_obj} 逐笔数据无'type'列，无法计算买卖量。")
                        buy_vol_per_minute = pd.Series(0, index=tick_df.index).resample('1min').sum()
                        sell_vol_per_minute = pd.Series(0, index=tick_df.index).resample('1min').sum()
                    else:
                        buy_vol_per_minute = tick_df[tick_df['type'] == 'B'].resample('1min')[current_volume_col].sum()
                        sell_vol_per_minute = tick_df[tick_df['type'] == 'S'].resample('1min')[current_volume_col].sum()
                    minute_df_from_ticks = tick_df.resample('1min').agg(
                        open=(current_price_col, 'first'), high=(current_price_col, 'max'), low=(current_price_col, 'min'),
                        close=(current_price_col, 'last'), vol=(current_volume_col, 'sum'), amount=(current_amount_col, 'sum')
                    ).dropna(subset=['open', 'high', 'low', 'close', 'vol', 'amount'])
                    minute_df_from_ticks['buy_vol_raw'] = buy_vol_per_minute
                    minute_df_from_ticks['sell_vol_raw'] = sell_vol_per_minute
                    minute_df_from_ticks.fillna(0, inplace=True)
                    intraday_data_map[date_obj] = self._group_minute_data_from_df(minute_df_from_ticks)
                    processed_with_tick_data = True
                except Exception as e:
                    logger.warning(f"[{stock_info.stock_code}] [资金流服务] 日期 {date_obj} 逐笔数据处理失败: {e}，将尝试回退到分钟数据。")
                    processed_with_tick_data = False
            if not processed_with_tick_data and minute_data_map and date_obj in minute_data_map:
                intraday_data_map[date_obj] = self._group_minute_data_from_df(minute_data_map[date_obj])
            elif not processed_with_tick_data:
                pass # 移除了此处的print调试信息
        return intraday_data_map

    def _calculate_all_metrics_for_day(self, stock_code: str, daily_data_series: pd.Series, intraday_data: pd.DataFrame, attributed_minute_df: pd.DataFrame, probabilistic_costs_dict: dict, tick_data_for_day: pd.DataFrame, level5_data_for_day: pd.DataFrame, realtime_data_for_day: pd.DataFrame, debug_mode: bool = False) -> tuple[dict, None]:
        day_metrics = {}
        daily_derived_metrics = self._calculate_daily_derived_metrics(daily_data_series, debug_mode=debug_mode)
        day_metrics.update(daily_derived_metrics)
        day_metrics.update(probabilistic_costs_dict)
        prob_costs_series = pd.Series(probabilistic_costs_dict)
        prob_costs_df_for_agg = pd.DataFrame([prob_costs_series], index=[daily_data_series.name])
        daily_df_for_agg = pd.DataFrame([daily_data_series.to_dict()], index=[daily_data_series.name])
        aggregate_pvwap_costs_df = self._calculate_aggregate_pvwap_costs(prob_costs_df_for_agg, daily_df_for_agg, debug_mode=debug_mode)
        if not aggregate_pvwap_costs_df.empty:
            day_metrics.update(aggregate_pvwap_costs_df.iloc[0].to_dict())
        updated_daily_data_series = pd.Series({**daily_data_series.to_dict(), **day_metrics}, name=daily_data_series.name)
        main_force_net_flow_calibrated = daily_derived_metrics.get('main_force_net_flow_calibrated')
        behavioral_metrics = self._compute_all_behavioral_metrics(
            stock_code, attributed_minute_df, updated_daily_data_series,
            tick_data=tick_data_for_day,
            level5_data=level5_data_for_day,
            realtime_data=realtime_data_for_day,
            main_force_net_flow_calibrated=main_force_net_flow_calibrated,
            debug_mode=debug_mode
        )
        day_metrics.update(behavioral_metrics)
        day_metrics['trade_time'] = daily_data_series.name
        return day_metrics, None

    def _synthesize_and_forge_metrics(self, stock_code: str, merged_df: pd.DataFrame, tick_data_map: dict = None, level5_data_map: dict = None, minute_data_map: dict = None, realtime_data_map: dict = None, memory: dict = None) -> tuple[pd.DataFrame, dict, list, dict]:
        all_metrics_list = []
        attributed_minute_data_map = {}
        failures = []
        prev_metrics = memory.copy() if memory is not None else {}
        num_days = len(merged_df)
        for i, (trade_date, daily_data_series) in enumerate(merged_df.iterrows()):
            debug_mode = (i == num_days - 1)
            date_obj = trade_date.date()
            daily_amount = pd.to_numeric(daily_data_series.get('amount'), errors='coerce') * 1000
            daily_vol_shares = pd.to_numeric(daily_data_series.get('vol'), errors='coerce') * 100
            if pd.notna(daily_amount) and pd.notna(daily_vol_shares) and daily_vol_shares > 0:
                daily_data_series['daily_vwap'] = daily_amount / daily_vol_shares
            else:
                daily_data_series['daily_vwap'] = np.nan
            intraday_data = self._minute_df_daily_grouped.get(date_obj)
            if intraday_data is None or intraday_data.empty:
                failures.append({'stock_code': stock_code, 'trade_date': str(date_obj), 'reason': '当日分钟线/逐笔聚合数据缺失'})
                continue
            daily_data_series_with_mem = pd.concat([daily_data_series, pd.Series(prev_metrics, name=daily_data_series.name)])
            attribution_weights_df = self._calculate_intraday_attribution_weights(intraday_data, daily_data_series_with_mem)
            probabilistic_costs_dict, attributed_minute_df = self._calculate_probabilistic_costs(stock_code, attribution_weights_df, daily_data_series_with_mem, debug_mode=debug_mode)
            day_metrics, _ = self._calculate_all_metrics_for_day(
                stock_code, daily_data_series_with_mem, intraday_data, attributed_minute_df, probabilistic_costs_dict,
                tick_data_for_day=tick_data_map.get(date_obj),
                level5_data_for_day=level5_data_map.get(date_obj),
                realtime_data_for_day=realtime_data_map.get(date_obj),
                debug_mode=debug_mode
            )
            all_metrics_list.append(day_metrics)
            attributed_minute_data_map[date_obj] = attributed_minute_df.copy(deep=True)
            next_prev_metrics = {
                'holistic_cmf': day_metrics.get('holistic_cmf'),
                'main_force_cmf': day_metrics.get('main_force_cmf'),
            }
            prev_metrics = next_prev_metrics
        if not all_metrics_list:
            return pd.DataFrame(), {}, failures, prev_metrics
        final_metrics_df = pd.DataFrame(all_metrics_list)
        final_metrics_df.set_index('trade_time', inplace=True)
        return final_metrics_df, attributed_minute_data_map, failures, prev_metrics

    def _calculate_daily_derived_metrics(self, daily_data_series: pd.Series, debug_mode: bool = False) -> dict:
        results = {}
        WAN = 10000.0
        def get_calibrated_value(target_col_name: str):
            consensus_map = {
                'net_flow_calibrated': ('net_flow_tushare', ['net_flow_ths', 'net_flow_dc']),
                'main_force_net_flow_calibrated': ('main_force_net_flow_tushare', ['main_force_net_flow_ths', 'main_force_net_flow_dc']),
                'retail_net_flow_calibrated': ('retail_net_flow_tushare', ['retail_net_flow_ths', 'retail_net_flow_dc']),
                'net_xl_amount_calibrated': ('net_xl_amount_tushare', ['net_xl_amount_dc']),
                'net_lg_amount_calibrated': ('net_lg_amount_tushare', ['net_lg_amount_ths', 'net_lg_amount_dc']),
                'net_md_amount_calibrated': ('net_md_amount_tushare', ['net_md_amount_ths', 'net_md_amount_dc']),
                'net_sh_amount_calibrated': ('net_sh_amount_tushare', ['net_sh_amount_ths', 'net_sh_amount_dc']),
            }
            if target_col_name not in consensus_map:
                return np.nan
            base_col, confirm_cols = consensus_map[target_col_name]
            base_value = pd.to_numeric(daily_data_series.get(base_col), errors='coerce')
            if pd.isna(base_value):
                for conf_col in confirm_cols:
                    alt_value = pd.to_numeric(daily_data_series.get(conf_col), errors='coerce')
                    if pd.notna(alt_value):
                        base_value = alt_value
                        break
            if pd.notna(base_value):
                confirmation_score = sum(1 for conf_col in confirm_cols if pd.notna(daily_data_series.get(conf_col)) and np.sign(base_value) == np.sign(pd.to_numeric(daily_data_series.get(conf_col), errors='coerce')))
                available_sources = sum(1 for conf_col in confirm_cols if pd.notna(daily_data_series.get(conf_col)))
                calibration_factor = (1 + confirmation_score) / (1 + available_sources) if available_sources > 0 else 1.0
                return base_value * calibration_factor
            return np.nan
        for col_name in ['net_flow_calibrated', 'main_force_net_flow_calibrated', 'retail_net_flow_calibrated', 'net_xl_amount_calibrated', 'net_lg_amount_calibrated', 'net_md_amount_calibrated', 'net_sh_amount_calibrated']:
            results[col_name] = get_calibrated_value(col_name)
        buy_sm = np.nan_to_num(pd.to_numeric(daily_data_series.get('buy_sm_amount'), errors='coerce'), nan=0.0)
        sell_sm = np.nan_to_num(pd.to_numeric(daily_data_series.get('sell_sm_amount'), errors='coerce'), nan=0.0)
        buy_md = np.nan_to_num(pd.to_numeric(daily_data_series.get('buy_md_amount'), errors='coerce'), nan=0.0)
        sell_md = np.nan_to_num(pd.to_numeric(daily_data_series.get('sell_md_amount'), errors='coerce'), nan=0.0)
        buy_lg = np.nan_to_num(pd.to_numeric(daily_data_series.get('buy_lg_amount'), errors='coerce'), nan=0.0)
        sell_lg = np.nan_to_num(pd.to_numeric(daily_data_series.get('sell_lg_amount'), errors='coerce'), nan=0.0)
        buy_elg = np.nan_to_num(pd.to_numeric(daily_data_series.get('buy_elg_amount'), errors='coerce'), nan=0.0)
        sell_elg = np.nan_to_num(pd.to_numeric(daily_data_series.get('sell_elg_amount'), errors='coerce'), nan=0.0)
        results['buy_sm_amount_calibrated'] = buy_sm
        results['sell_sm_amount_calibrated'] = sell_sm
        results['buy_md_amount_calibrated'] = buy_md
        results['sell_md_amount_calibrated'] = sell_md
        results['buy_lg_amount_calibrated'] = buy_lg
        results['sell_lg_amount_calibrated'] = sell_lg
        results['buy_elg_amount_calibrated'] = buy_elg
        results['sell_elg_amount_calibrated'] = sell_elg
        results['total_buy_amount_calibrated'] = buy_sm + buy_md + buy_lg + buy_elg
        results['total_sell_amount_calibrated'] = sell_sm + sell_md + sell_lg + sell_elg
        results['main_force_buy_amount_calibrated'] = buy_lg + buy_elg
        results['main_force_sell_amount_calibrated'] = sell_lg + sell_elg
        results['retail_buy_amount_calibrated'] = buy_sm + buy_md
        results['retail_sell_amount_calibrated'] = sell_sm + sell_md
        turnover_amount_yuan = pd.to_numeric(daily_data_series.get('amount'), errors='coerce') * 1000
        try:
            if turnover_amount_yuan > 0:
                base_flow_yuan = pd.to_numeric(daily_data_series.get('main_force_net_flow_tushare'), errors='coerce') * WAN
                confirm_flows_yuan = [pd.to_numeric(daily_data_series.get(c), errors='coerce') * WAN for c in ['main_force_net_flow_ths', 'main_force_net_flow_dc']]
                if pd.notna(base_flow_yuan):
                    deviations = [abs(conf_flow - base_flow_yuan) / turnover_amount_yuan for conf_flow in confirm_flows_yuan if pd.notna(conf_flow)]
                    results['flow_credibility_index'] = (1.0 - np.mean(deviations)) * 100 if deviations else 50.0
            else:
                results['flow_credibility_index'] = np.nan
        except Exception:
            results['flow_credibility_index'] = np.nan
        try:
            mf_flow_calibrated = results.get('main_force_net_flow_calibrated')
            retail_flow_calibrated = results.get('retail_net_flow_calibrated')
            if turnover_amount_yuan > 0 and pd.notna(mf_flow_calibrated) and pd.notna(retail_flow_calibrated):
                mf_flow_yuan = mf_flow_calibrated * WAN
                retail_flow_yuan = retail_flow_calibrated * WAN
                battle_volume_yuan = min(abs(mf_flow_yuan), abs(retail_flow_yuan))
                battle_turnover_yuan = 2 * battle_volume_yuan
                results['mf_retail_battle_intensity'] = (battle_turnover_yuan / turnover_amount_yuan) * 100
            else:
                results['mf_retail_battle_intensity'] = np.nan
        except Exception:
            results['mf_retail_battle_intensity'] = np.nan
        try:
            mf_flow_calibrated = results.get('main_force_net_flow_calibrated')
            retail_flow_calibrated = results.get('retail_net_flow_calibrated')
            if pd.notna(mf_flow_calibrated) and pd.notna(retail_flow_calibrated):
                mf_flow_yuan = mf_flow_calibrated * WAN
                retail_flow_yuan = retail_flow_calibrated * WAN
                total_opinionated_flow_yuan = abs(mf_flow_yuan) + abs(retail_flow_yuan)
                if total_opinionated_flow_yuan > 0:
                    dominance_ratio = abs(retail_flow_yuan) / total_opinionated_flow_yuan
                    divergence_penalty = 1 if np.sign(mf_flow_yuan) != np.sign(retail_flow_yuan) and mf_flow_yuan != 0 and retail_flow_yuan != 0 else 0
                    results['retail_flow_dominance_index'] = np.sign(retail_flow_yuan) * dominance_ratio * (1 + divergence_penalty) * 100
                else:
                    results['retail_flow_dominance_index'] = np.nan
            else:
                results['retail_flow_dominance_index'] = np.nan
        except Exception:
            results['retail_flow_dominance_index'] = np.nan
        return results

    def _calculate_probabilistic_costs(self, stock_code: str, minute_data_for_day: pd.DataFrame, daily_data: pd.Series, debug_mode: bool = False) -> tuple[dict, pd.DataFrame]:
        """
        【V6.15 · 诊断探针植入版】
        - 核心增强: 植入诊断探针，用于在debug模式下打印计算概率成本的关键输入与输出，定位成本指标计算失败的根源。
        """
        if minute_data_for_day is None or minute_data_for_day.empty:
            return {}, pd.DataFrame()
        day_results = {}
        cost_types = ['sm_buy', 'sm_sell', 'md_buy', 'md_sell', 'lg_buy', 'lg_sell', 'elg_buy', 'elg_sell']
        df_to_attribute = minute_data_for_day
        # 移除了所有与debug_mode和探针相关的print语句
        for cost_type in cost_types:
            size, direction = cost_type.split('_')
            db_vol_key = f'{direction}_{size}_vol'
            daily_vol_shares = pd.to_numeric(daily_data.get(db_vol_key), errors='coerce') * 100
            if pd.isna(daily_vol_shares) or daily_vol_shares == 0:
                day_results[f'avg_cost_{cost_type}'] = np.nan
                df_to_attribute[f'{cost_type}_vol_attr'] = 0
                continue
            weight_col = f'{size}_{direction}_weight'
            if weight_col not in df_to_attribute.columns:
                day_results[f'avg_cost_{cost_type}'] = np.nan
                df_to_attribute[f'{cost_type}_vol_attr'] = 0
                continue
            weight_series = df_to_attribute[weight_col]
            if weight_series.sum() < 1e-9:
                day_results[f'avg_cost_{cost_type}'] = np.nan
                df_to_attribute[f'{cost_type}_vol_attr'] = 0
                continue
            attributed_vol = weight_series * daily_vol_shares
            df_to_attribute[f'{cost_type}_vol_attr'] = attributed_vol
            attributed_value = attributed_vol * df_to_attribute['minute_vwap']
            total_attributed_value = attributed_value.sum()
            total_attributed_vol = attributed_vol.sum()
            calculated_cost = total_attributed_value / total_attributed_vol if total_attributed_vol > 0 else np.nan
            day_results[f'avg_cost_{cost_type}'] = calculated_cost
        fully_attributed_df = self._attribute_minute_volume_to_players(df_to_attribute)
        return day_results, fully_attributed_df

    def _calculate_aggregate_pvwap_costs(self, pvwap_df: pd.DataFrame, daily_df: pd.DataFrame, debug_mode: bool = False) -> pd.DataFrame:
        """
        【V49.2 · 执行力穿透版】
        - 核心修复: 修正了此前版本中，已计算的聚合成本（avg_cost_main_buy/sell）未被包含在返回结果中的致命缺陷。
                     现在，这些关键的中间成本被正确地添加到返回的DataFrame中，从而打通了整个计算链路的“最后一公里”，
                     确保下游指标（如retail_fomo_premium_index）能够获取到它们所依赖的数据。
        - 核心重构: 移除了 main_force_execution_alpha 和 main_force_t0_efficiency 的计算逻辑，
                     将其职责转移至新的 _calculate_execution_alpha_metrics 方法，实现单一职责原则。
        """
        if pvwap_df.empty or daily_df.empty:
            return pd.DataFrame()
        temp_df = pvwap_df.copy()
        result_agg_df = pd.DataFrame(index=pvwap_df.index)
        def weighted_average_cost(cost_keys, vol_keys):
            total_value = 0
            total_volume = 0
            for cost_key, vol_key in zip(cost_keys, vol_keys):
                cost = pd.to_numeric(temp_df.get(cost_key, np.nan).iloc[0], errors='coerce')
                vol = pd.to_numeric(daily_df.get(vol_key, 0).iloc[0], errors='coerce') * 100
                if pd.notna(cost) and pd.notna(vol) and vol > 0:
                    total_value += cost * vol
                    total_volume += vol
            return total_value / total_volume if total_volume > 0 else np.nan
        temp_df['avg_cost_main_buy'] = weighted_average_cost(['avg_cost_lg_buy', 'avg_cost_elg_buy'], ['buy_lg_vol', 'buy_elg_vol'])
        temp_df['avg_cost_main_sell'] = weighted_average_cost(['avg_cost_lg_sell', 'avg_cost_elg_sell'], ['sell_lg_vol', 'sell_elg_vol'])
        temp_df['avg_cost_retail_buy'] = weighted_average_cost(['avg_cost_sm_buy', 'avg_cost_md_buy'], ['buy_sm_vol', 'buy_md_vol'])
        temp_df['avg_cost_retail_sell'] = weighted_average_cost(['avg_cost_sm_sell', 'avg_cost_md_sell'], ['sell_sm_vol', 'sell_md_vol'])
        result_agg_df['avg_cost_main_buy'] = temp_df['avg_cost_main_buy']
        result_agg_df['avg_cost_main_sell'] = temp_df['avg_cost_main_sell']
        result_agg_df['avg_cost_retail_buy'] = temp_df['avg_cost_retail_buy']
        result_agg_df['avg_cost_retail_sell'] = temp_df['avg_cost_retail_sell']
        temp_df['daily_vwap'] = daily_df['daily_vwap']
        temp_df['atr_14d'] = daily_df['atr_14d']
        try:
            alpha = (temp_df['avg_cost_main_buy'] - temp_df['avg_cost_main_sell']) / temp_df['daily_vwap']
            result_agg_df['main_force_cost_alpha'] = alpha * 100
        except Exception:
            result_agg_df['main_force_cost_alpha'] = np.nan
        try:
            beta = (temp_df['avg_cost_retail_buy'] - temp_df['avg_cost_retail_sell']) / temp_df['daily_vwap']
            result_agg_df['retail_cost_beta'] = beta * 100
        except Exception:
            result_agg_df['retail_cost_beta'] = np.nan
        # 移除 main_force_t0_spread_ratio, main_force_execution_alpha, main_force_t0_efficiency 的计算逻辑
        try:
            mf_cost_premium = (temp_df['avg_cost_main_buy'] / temp_df['daily_vwap'] - 1)
            retail_cost_discount = (1 - temp_df['avg_cost_retail_sell'] / temp_df['daily_vwap'])
            temperature = mf_cost_premium - retail_cost_discount
            result_agg_df['flow_temperature_premium'] = temperature * 100
        except Exception:
            result_agg_df['flow_temperature_premium'] = np.nan
        return result_agg_df

    def _attribute_minute_volume_to_players(self, minute_df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.1】将基础成交量归因为主力/散户的核心辅助函数。
        - 核心职责: 聚合基础的 *_vol_attr 列，生成 main_force_* 和 retail_* 级别的成交量列。
        """
        df = minute_df.copy()
        # 移除了所有与debug_params和probe_dates相关的探针初始化代码
        df['main_force_buy_vol'] = df.get('lg_buy_vol_attr', 0) + df.get('elg_buy_vol_attr', 0)
        df['main_force_sell_vol'] = df.get('lg_sell_vol_attr', 0) + df.get('elg_sell_vol_attr', 0)
        df['main_force_net_vol'] = df['main_force_buy_vol'] - df['main_force_sell_vol']
        df['retail_buy_vol'] = df.get('sm_buy_vol_attr', 0) + df.get('md_buy_vol_attr', 0)
        df['retail_sell_vol'] = df.get('sm_sell_vol_attr', 0) + df.get('md_sell_vol_attr', 0)
        df['retail_net_vol'] = df['retail_buy_vol'] - df['retail_sell_vol']
        # 移除了检查归因后成交量的探针print语句
        return df

    def _calculate_derivatives(self, stock_code: str, consensus_df: pd.DataFrame) -> pd.DataFrame:
        derivatives_df = pd.DataFrame(index=consensus_df.index)
        import pandas_ta as ta
        # 移除了所有与debug_params和probe_dates相关的探针初始化代码
        SLOPE_ACCEL_EXCLUSIONS = BaseAdvancedFundFlowMetrics.SLOPE_ACCEL_EXCLUSIONS
        CORE_METRICS_TO_DERIVE = list(BaseAdvancedFundFlowMetrics.CORE_METRICS.keys())
        ACCEL_WINDOW = 2
        sum_cols = [
            'net_flow_calibrated', 'main_force_net_flow_calibrated', 'retail_net_flow_calibrated',
            'net_xl_amount_calibrated', 'net_lg_amount_calibrated', 'net_md_amount_calibrated',
            'net_sh_amount_calibrated', 'main_force_on_peak_flow',
        ]
        UNIFIED_PERIODS = [1, 5, 13, 21, 55]
        for p in UNIFIED_PERIODS:
            if p <= 1: continue
            min_p = max(2, int(p * 0.8))
            for col in sum_cols:
                if col in consensus_df.columns:
                    source_series_for_sum = pd.to_numeric(consensus_df[col], errors='coerce')
                    # 移除了检查数据源缺失值的探针print语句
                    sum_col_name = f'{col}_sum_{p}d'
                    derivatives_df[sum_col_name] = source_series_for_sum.rolling(window=p, min_periods=min_p).sum()
                else:
                    pass # 移除了检查数据源列是否存在的探针print语句
        all_cols_to_derive = CORE_METRICS_TO_DERIVE + list(derivatives_df.columns)
        for col in all_cols_to_derive:
            base_col_name = col.split('_sum_')[0] if '_sum_' in col else col
            if base_col_name in SLOPE_ACCEL_EXCLUSIONS:
                continue
            if col in consensus_df.columns:
                source_series = pd.to_numeric(consensus_df[col], errors='coerce')
            elif col in derivatives_df.columns:
                source_series = derivatives_df[col]
            else:
                continue
            # 移除了检查数据源是否全为缺失值的探针print语句
            if source_series.isnull().all():
                continue
            for p in UNIFIED_PERIODS:
                calc_window = max(2, p)
                slope_col_name = f'{col}_slope_{p}d'
                slope_series = ta.slope(close=source_series.astype(float), length=calc_window)
                derivatives_df[slope_col_name] = slope_series
                if slope_series is not None and not slope_series.empty:
                    accel_col_name = f'{col}_accel_{p}d'
                    derivatives_df[accel_col_name] = ta.slope(close=slope_series.astype(float), length=ACCEL_WINDOW)
        return derivatives_df

    def _calculate_advanced_behavioral_metrics(self, daily_df: pd.DataFrame, minute_df_attributed_grouped: dict) -> pd.DataFrame:
        """
        【V28.0 · 行为计算核心整合版】
        - 核心重构: 废弃所有零散的行为计算方法，引入统一的计算引擎 `_compute_all_behavioral_metrics`。
        - 核心思想: 本方法负责数据准备与调度，将所有分钟级行为指标的计算逻辑内聚到单一引擎中。
        """
        if not minute_df_attributed_grouped:
            return pd.DataFrame(index=daily_df.index)
        all_results = {}
        for date, daily_data in daily_df.iterrows():
            if date not in minute_df_attributed_grouped:
                continue
            minute_data = minute_df_attributed_grouped[date].copy()
            if minute_data.empty:
                continue
            # 调用统一计算引擎
            day_results = self._compute_all_behavioral_metrics(minute_data, daily_data)
            day_results['trade_time'] = date
            all_results[date] = day_results
        if not all_results:
            return pd.DataFrame()
        return pd.DataFrame.from_dict(all_results, orient='index').set_index('trade_time')

    def _compute_all_behavioral_metrics(self, stock_code: str, intraday_data: pd.DataFrame, daily_data: pd.Series, tick_data: pd.DataFrame = None, level5_data: pd.DataFrame = None, realtime_data: pd.DataFrame = None, main_force_net_flow_calibrated: float = None, debug_mode: bool = False) -> dict:
        """
        【V1.5 · 主力动态识别版】计算所有行为指标的统一入口。
        - 核心升级: 继承类级别的主力动态识别逻辑，确保所有依赖主力交易的指标都基于更精确的定义。
        """
        results = {}
        if intraday_data.empty:
            return results
        raw_hf_df, common_data = self._prepare_behavioral_data(
            intraday_data, daily_data, tick_data, level5_data, realtime_data
        )
        current_date = daily_data.name.date()
        should_probe = self.debug_params.get('should_probe', False) and \
                       (current_date.strftime('%Y-%m-%d') in self.debug_params.get('probe_dates', []))
        context = {
            'intraday_data': intraday_data,
            'daily_data': daily_data,
            'hf_analysis_df': None,
            'common_data': common_data,
            'hf_features': None,
            'main_force_net_flow_calibrated': main_force_net_flow_calibrated,
            'debug': {
                'should_probe': should_probe,
                'probe_dates': self.debug_params.get('probe_dates', []),
                'stock_code': stock_code
            }
        }
        hf_analysis_df, hf_features = self._engineer_hf_features(raw_hf_df, common_data.get('daily_total_volume', 0), context)
        context['hf_analysis_df'] = hf_analysis_df
        context['hf_features'] = hf_features
        mf_metrics = AdvancedFundFlowMetricsService._calculate_level5_order_flow_metrics(context)
        results.update(mf_metrics)
        context['mf_metrics'] = mf_metrics
        if not hf_analysis_df.empty:
            results.update(AdvancedFundFlowMetricsService._calculate_main_force_profile_metrics(context))
            results.update(AdvancedFundFlowMetricsService._calculate_ofi_based_metrics(context))
            results.update(AdvancedFundFlowMetricsService._calculate_order_book_metrics(context))
            results.update(AdvancedFundFlowMetricsService._calculate_micro_dynamics_metrics(context))
            results.update(AdvancedFundFlowMetricsService._calculate_main_force_intraday_intent(context))
            main_force_daily_buy_amount = hf_features.get('main_force_daily_buy_amount', 0.0)
            main_force_daily_sell_amount = hf_features.get('main_force_daily_sell_amount', 0.0)
            main_force_daily_buy_volume = hf_features.get('main_force_daily_buy_volume', 0.0)
            main_force_daily_sell_volume = hf_features.get('main_force_daily_sell_volume', 0.0)
            results['main_force_daily_buy_amount_D'] = main_force_daily_buy_amount
            results['main_force_daily_sell_amount_D'] = main_force_daily_sell_amount
            results['main_force_net_amount_from_hf_D'] = main_force_daily_buy_amount - main_force_daily_sell_amount
            results['main_force_daily_buy_volume_D'] = main_force_daily_buy_volume
            results['main_force_daily_sell_volume_D'] = main_force_daily_sell_volume
            results['main_force_net_volume_from_hf_D'] = main_force_daily_buy_volume - main_force_daily_sell_volume
            # if should_probe and current_date.strftime('%Y-%m-%d') in context['debug']['probe_dates']:
            #     print(f"\n--- [探针 _compute_all_behavioral_metrics] {stock_code} {current_date} - 主力日度净买卖金额/股数结果 ---")
            #     print(f"  - 从 hf_features 获取:")
            #     print(f"    - main_force_aggressive_buy_volume: {hf_features.get('main_force_aggressive_buy_volume', 0.0):.2f}")
            #     print(f"    - main_force_aggressive_sell_volume: {hf_features.get('main_force_aggressive_sell_volume', 0.0):.2f}")
            #     print(f"    - main_force_passive_buy_volume: {hf_features.get('main_force_passive_buy_volume', 0.0):.2f}")
            #     print(f"    - main_force_passive_sell_volume: {hf_features.get('main_force_passive_sell_volume', 0.0):.2f}")
            #     print(f"    - main_force_daily_buy_amount: {main_force_daily_buy_amount:.2f}")
            #     print(f"    - main_force_daily_sell_amount: {main_force_daily_sell_amount:.2f}")
            #     print(f"    - main_force_daily_buy_volume: {main_force_daily_buy_volume:.2f}")
            #     print(f"    - main_force_daily_sell_volume: {main_force_daily_sell_volume:.2f}")
            #     print(f"  - 最终结果 (results):")
            #     print(f"    - main_force_daily_buy_amount_D: {results['main_force_daily_buy_amount_D']:.2f}")
            #     print(f"    - main_force_daily_sell_amount_D: {results['main_force_daily_sell_amount_D']:.2f}")
            #     print(f"    - main_force_net_amount_from_hf_D: {results['main_force_net_amount_from_hf_D']:.2f}")
            #     print(f"    - main_force_daily_buy_volume_D: {results['main_force_daily_buy_volume_D']:.2f}")
            #     print(f"    - main_force_daily_sell_volume_D: {results['main_force_daily_sell_volume_D']:.2f}")
            #     print(f"    - main_force_net_volume_from_hf_D: {results['main_force_net_volume_from_hf_D']:.2f}")
            #     print(f"--- [探针 _compute_all_behavioral_metrics 结束] {stock_code} {current_date} ---")
        results.update(AdvancedFundFlowMetricsService._calculate_vwap_related_metrics(context))
        results.update(AdvancedFundFlowMetricsService._calculate_vwap_control_metrics(context))
        results.update(AdvancedFundFlowMetricsService._calculate_opening_battle_metrics(context))
        results.update(AdvancedFundFlowMetricsService._calculate_shadow_metrics(context))
        results.update(AdvancedFundFlowMetricsService._calculate_dip_rally_metrics(context, raw_hf_df))
        results.update(AdvancedFundFlowMetricsService._calculate_reversal_metrics(context))
        results.update(AdvancedFundFlowMetricsService._calculate_panic_cascade_metrics(context))
        results.update(AdvancedFundFlowMetricsService._calculate_cmf_metrics(context))
        results.update(AdvancedFundFlowMetricsService._calculate_vpoc_metrics(context))
        results.update(AdvancedFundFlowMetricsService._calculate_liquidity_swap_metrics(context))
        results.update(AdvancedFundFlowMetricsService._calculate_closing_metrics(context))
        results.update(AdvancedFundFlowMetricsService._calculate_retail_sentiment_metrics(context))
        results.update(AdvancedFundFlowMetricsService._calculate_hidden_accumulation_metrics(context))
        results.update(AdvancedFundFlowMetricsService._calculate_execution_alpha_metrics(context))
        results.update(AdvancedFundFlowMetricsService._calculate_flow_efficiency_metrics(context))
        results.update(AdvancedFundFlowMetricsService._calculate_wash_trade_metrics(context))
        results.update(AdvancedFundFlowMetricsService._calculate_misc_minute_metrics(context))
        results.update(AdvancedFundFlowMetricsService._calculate_closing_strength_metrics(context))
        results.update(AdvancedFundFlowMetricsService._calculate_misc_daily_metrics(context))
        return results

    @staticmethod
    def _identify_trade_participants(hf_analysis_df: pd.DataFrame, context: dict) -> Tuple[pd.Series, pd.Series]:
        """
        精确识别高频交易数据中的主力交易和散户交易。
        重构逻辑：基于多维度微观市场结构特征进行精细化识别。
        参数:
            hf_analysis_df (pd.DataFrame): 包含高频交易数据和Level5盘口数据的DataFrame
            context (dict): 包含调试信息等上下文的字典。
        返回:
            Tuple[pd.Series, pd.Series]:
                - is_main_force_trade (pd.Series): 布尔Series，标记是否为主力交易
                - is_retail_trade (pd.Series): 布尔Series，标记是否为散户交易
        """
        # 获取调试信息
        should_probe = context['debug'].get('should_probe', False)
        stock_code = context['debug']['stock_code']
        current_date = context['daily_data'].name.date()

        if hf_analysis_df.empty:
            empty_mask = pd.Series(False, index=hf_analysis_df.index, dtype=bool)
            if should_probe:
                print(f"\n--- [探针 _identify_trade_participants] {stock_code} {current_date} ---")
                print(f"  - hf_analysis_df is empty, returning empty masks.")
            return empty_mask, empty_mask
        
        # 探针：检查输入数据
        if should_probe:
            print(f"\n--- [探针 _identify_trade_participants - 输入数据检查] {stock_code} {current_date} ---")
            print(f"  - hf_analysis_df shape: {hf_analysis_df.shape}")
            # 尝试获取索引频率，如果索引不是DatetimeIndex，则可能没有freq属性
            index_freq = None
            if isinstance(hf_analysis_df.index, pd.DatetimeIndex):
                index_freq = pd.infer_freq(hf_analysis_df.index)
            print(f"  - hf_analysis_df index frequency: {index_freq}")
            print(f"  - hf_analysis_df['amount'] describe:\n{hf_analysis_df['amount'].describe()}")
            print(f"  - hf_analysis_df['volume'] describe:\n{hf_analysis_df['volume'].describe()}")
            # 打印前5行，避免输出过长
            print(f"  - hf_analysis_df head:\n{hf_analysis_df.head().to_string()}")

        # 基础参数设置
        # 散户交易的最大金额阈值，低于此金额的交易是散户的候选
        RETAIL_MAX_AMOUNT = 50000 
        # 主力交易的最小金额阈值，高于此金额的交易是主力的强力候选
        MAIN_FORCE_MIN_AMOUNT = 200000 
        # 用于某些主力特征的最小成交量阈值
        MIN_ABSOLUTE_VOLUME_THRESHOLD = 5000 
        
        # 1. 基础金额和成交量分析
        avg_trade_amount = hf_analysis_df['amount'].mean()
        avg_trade_volume = hf_analysis_df['volume'].mean()
        
        # 2. 微观市场结构特征计算 (保持不变)
        # 2.1 盘口压力指标
        # 确保所有必要的列都存在，否则跳过相关计算
        if all(col in hf_analysis_df.columns for col in ['buy_volume1', 'sell_volume1', 'buy_price1', 'sell_price1']):
            hf_analysis_df['bid_ask_pressure_ratio'] = hf_analysis_df['buy_volume1'] / (hf_analysis_df['sell_volume1'] + 1e-10)
            hf_analysis_df['mid_price'] = (hf_analysis_df['buy_price1'] + hf_analysis_df['sell_price1']) / 2
            hf_analysis_df['price_impact'] = (hf_analysis_df['price'] - hf_analysis_df['mid_price']) / hf_analysis_df['mid_price']
            if 'prev_a1_v' in hf_analysis_df.columns and 'prev_b1_v' in hf_analysis_df.columns:
                hf_analysis_df['ask_depth_change'] = (hf_analysis_df['sell_volume1'] - hf_analysis_df['prev_a1_v']) / (hf_analysis_df['prev_a1_v'] + 1e-10)
                hf_analysis_df['bid_depth_change'] = (hf_analysis_df['buy_volume1'] - hf_analysis_df['prev_b1_v']) / (hf_analysis_df['prev_b1_v'] + 1e-10)
            else:
                hf_analysis_df['ask_depth_change'] = np.nan
                hf_analysis_df['bid_depth_change'] = np.nan
        else:
            # 如果缺少关键列，则将相关特征初始化为默认值
            hf_analysis_df['bid_ask_pressure_ratio'] = np.nan
            hf_analysis_df['mid_price'] = np.nan
            hf_analysis_df['price_impact'] = np.nan
            hf_analysis_df['ask_depth_change'] = np.nan
            hf_analysis_df['bid_depth_change'] = np.nan

        # 2.2 订单流特征
        if 'type' in hf_analysis_df.columns and 'sell_price1' in hf_analysis_df.columns and 'buy_price1' in hf_analysis_df.columns:
            aggressive_buy_mask = (hf_analysis_df['type'] == 'B') & (hf_analysis_df['price'] >= hf_analysis_df['sell_price1'] * 0.999)
            aggressive_sell_mask = (hf_analysis_df['type'] == 'S') & (hf_analysis_df['price'] <= hf_analysis_df['buy_price1'] * 1.001)
            hf_analysis_df['aggressive_buy_pressure'] = aggressive_buy_mask.astype(int)
            hf_analysis_df['aggressive_sell_pressure'] = aggressive_sell_mask.astype(int)
        else:
            hf_analysis_df['aggressive_buy_pressure'] = 0
            hf_analysis_df['aggressive_sell_pressure'] = 0
        
        # 2.3 价格动量特征
        if 'price' in hf_analysis_df.columns:
            hf_analysis_df['price_sma_5'] = hf_analysis_df['price'].rolling(window=5, min_periods=1).mean()
            hf_analysis_df['price_momentum'] = (hf_analysis_df['price'] - hf_analysis_df['price_sma_5']) / hf_analysis_df['price_sma_5']
        else:
            hf_analysis_df['price_momentum'] = np.nan
        
        # --- 3. 散户交易识别逻辑 (优先识别) ---
        is_retail_trade = pd.Series(False, index=hf_analysis_df.index, dtype=bool)
        
        # 3.1 基础条件：金额小于 RETAIL_MAX_AMOUNT
        small_amount_mask = hf_analysis_df['amount'] < RETAIL_MAX_AMOUNT
        if should_probe:
            print(f"  - small_amount_mask count (amount < {RETAIL_MAX_AMOUNT}): {small_amount_mask.sum()}")
        
        # 3.2 微观特征：无显著价格冲击、盘口压力正常
        normal_microstructure = pd.Series(True, index=hf_analysis_df.index, dtype=bool)
        if 'price_impact' in hf_analysis_df.columns:
            normal_microstructure = normal_microstructure & (hf_analysis_df['price_impact'].abs() <= 0.0005)  # 价格冲击小于0.05%
        if 'bid_ask_pressure_ratio' in hf_analysis_df.columns:
            normal_microstructure = normal_microstructure & (hf_analysis_df['bid_ask_pressure_ratio'] >= 0.8) & \
                                                         (hf_analysis_df['bid_ask_pressure_ratio'] <= 1.2)  # 盘口压力平衡
        
        # 3.3 交易模式：非攻击性交易 (中间价成交或被动成交)
        non_aggressive_trade = pd.Series(True, index=hf_analysis_df.index, dtype=bool)
        if 'type' in hf_analysis_df.columns and 'price' in hf_analysis_df.columns and 'sell_price1' in hf_analysis_df.columns and 'buy_price1' in hf_analysis_df.columns:
            mid_price_trade = (hf_analysis_df['type'] == 'M') | \
                             ((hf_analysis_df['price'] > hf_analysis_df['buy_price1'] * 1.001) & \
                              (hf_analysis_df['price'] < hf_analysis_df['sell_price1'] * 0.999))
            non_aggressive_trade = non_aggressive_trade & mid_price_trade
        
        # 3.4 综合判断散户交易：必须满足金额小、微观结构正常、非攻击性
        is_retail_trade = small_amount_mask & normal_microstructure & non_aggressive_trade
        if should_probe:
            print(f"  - is_retail_trade count (after all retail conditions): {is_retail_trade.sum()}")
        
        # --- 4. 主力交易识别逻辑 (在排除散户后进行) ---
        is_main_force_trade = pd.Series(False, index=hf_analysis_df.index, dtype=bool)
        
        # 4.1 绝对金额阈值 - 大额交易直接认定为主力
        absolute_amount_mask = hf_analysis_df['amount'] >= MAIN_FORCE_MIN_AMOUNT
        if should_probe:
            print(f"  - absolute_amount_mask count (amount >= {MAIN_FORCE_MIN_AMOUNT}): {absolute_amount_mask.sum()}")
        
        # 4.2 相对金额和成交量异常 - 显著高于平均水平 (适用于非散户交易)
        amount_multiplier_threshold = 5.0
        volume_multiplier_threshold = 5.0
        relative_anomaly_mask = (hf_analysis_df['amount'] > amount_multiplier_threshold * avg_trade_amount) | \
                               (hf_analysis_df['volume'] > volume_multiplier_threshold * avg_trade_volume)
        
        # 4.3 微观结构特征识别 - 基于盘口压力和大单行为 (适用于非散户交易)
        microstructure_mask = pd.Series(False, index=hf_analysis_df.index, dtype=bool)
        if 'bid_ask_pressure_ratio' in hf_analysis_df.columns and 'price_impact' in hf_analysis_df.columns:
            # 条件1：金额大于 RETAIL_MAX_AMOUNT 且伴随显著的价格冲击
            large_trade_with_impact = (hf_analysis_df['amount'] >= RETAIL_MAX_AMOUNT) & \
                                     (hf_analysis_df['price_impact'].abs() > 0.001)
            # 条件2：金额大于 RETAIL_MAX_AMOUNT 且异常盘口压力下的交易
            abnormal_pressure_trade = (hf_analysis_df['amount'] >= RETAIL_MAX_AMOUNT) & \
                                     ((hf_analysis_df['bid_ask_pressure_ratio'] > 2.0) | (hf_analysis_df['bid_ask_pressure_ratio'] < 0.5))
            # 条件3：金额大于 RETAIL_MAX_AMOUNT 且挂单深度显著变化时的交易
            if 'ask_depth_change' in hf_analysis_df.columns and 'bid_depth_change' in hf_analysis_df.columns:
                depth_change_trade = (hf_analysis_df['amount'] >= RETAIL_MAX_AMOUNT) & \
                                    ((hf_analysis_df['ask_depth_change'].abs() > 0.3) | (hf_analysis_df['bid_depth_change'].abs() > 0.3))
                microstructure_mask = large_trade_with_impact | abnormal_pressure_trade | depth_change_trade
            else:
                microstructure_mask = large_trade_with_impact | abnormal_pressure_trade
        
        # 4.4 订单流特征识别 - 主动攻击性大单 (适用于非散户交易)
        order_flow_mask = pd.Series(False, index=hf_analysis_df.index, dtype=bool)
        if 'aggressive_buy_pressure' in hf_analysis_df.columns and 'aggressive_sell_pressure' in hf_analysis_df.columns:
            # 主动攻击性且金额大于 RETAIL_MAX_AMOUNT
            aggressive_large_trade = (hf_analysis_df['amount'] >= RETAIL_MAX_AMOUNT) & \
                                    ((hf_analysis_df['aggressive_buy_pressure'] == 1) | (hf_analysis_df['aggressive_sell_pressure'] == 1))
            # 动量跟随且金额大于 RETAIL_MAX_AMOUNT
            if 'price_momentum' in hf_analysis_df.columns:
                momentum_follow_trade = (hf_analysis_df['amount'] >= RETAIL_MAX_AMOUNT) & \
                                       (hf_analysis_df['price_momentum'].abs() > 0.001) & \
                                       (hf_analysis_df['volume'] >= MIN_ABSOLUTE_VOLUME_THRESHOLD)
                order_flow_mask = aggressive_large_trade | momentum_follow_trade
            else:
                order_flow_mask = aggressive_large_trade
        
        # 4.5 综合判断主力交易 - 满足任一条件即可，且不能是已识别的散户交易
        is_main_force_trade = (absolute_amount_mask | relative_anomaly_mask | microstructure_mask | order_flow_mask) & (~is_retail_trade)
        if should_probe:
            print(f"  - is_main_force_trade count (after all main force conditions and excluding retail): {is_main_force_trade.sum()}")
            print(f"--- [探针 _identify_trade_participants 结束] ---")
        
        return is_main_force_trade, is_retail_trade

    @staticmethod
    def _calculate_main_force_profile_metrics(context: dict) -> dict:
        """
        【V69.1 · 韧性升维版 - 空数据鲁棒性增强】
        - 核心逻辑: `main_force_conviction_index` 的“韧性”组件基于 `mid_price_change < 0` (实际价格下跌) 计算，
                     以更精确地衡量主力在真实逆境中的托底决心。
        - 核心增强: 增加对 `hf_analysis_df` 是否为空的检查，避免在无高频数据时引发 `KeyError`。
        - 核心修复: 将所有对 `mid_price_delta` 的引用改为 `mid_price_change`，以保持命名一致性。
        """
        hf_analysis_df = context['hf_analysis_df']
        common_data = context['common_data']
        hf_features = context['hf_features']
        should_probe = context['debug']['should_probe']
        stock_code = context['debug']['stock_code']
        current_date = context['daily_data'].name.date()
        probe_active = should_probe and (current_date.strftime('%Y-%m-%d') in context['debug']['probe_dates'])
        import numpy as np
        metrics = {}
        if hf_analysis_df.empty:
            return metrics
        atr = common_data['atr']
        daily_total_volume = common_data['daily_total_volume']
        mf_ofi_cumsum = hf_analysis_df['main_force_ofi'].cumsum().fillna(0)
        aggressiveness_component, trend_quality, closing_strength = 0.0, 0.0, 0.0
        if not mf_ofi_cumsum.empty and mf_ofi_cumsum.nunique() > 1:
            time_index = np.arange(len(mf_ofi_cumsum))
            if len(time_index) > 1 and mf_ofi_cumsum.nunique() > 1:
                trend_quality = np.corrcoef(time_index, mf_ofi_cumsum)[0, 1]
                trend_quality = np.nan_to_num(trend_quality)
            else:
                trend_quality = 0.0
            ofi_min, ofi_max = mf_ofi_cumsum.min(), mf_ofi_cumsum.max()
            if (ofi_max - ofi_min) > 0:
                closing_strength = (mf_ofi_cumsum.iloc[-1] - ofi_min) / (ofi_max - ofi_min)
            else:
                closing_strength = 0.0
            closing_strength = np.nan_to_num(closing_strength)
            aggressiveness_component = trend_quality * closing_strength
        cost_tolerance_component = 0.0
        hf_mf_buy_vwap = hf_features.get('hf_mf_buy_vwap')
        hf_mf_sell_vwap = hf_features.get('hf_mf_sell_vwap')
        if pd.notna(hf_mf_buy_vwap) and pd.notna(hf_mf_sell_vwap) and pd.notna(atr) and atr > 0:
            cost_tolerance_component = (hf_mf_buy_vwap - hf_mf_sell_vwap) / atr
        resilience_component = 0.0
        if 'mid_price_change' in hf_analysis_df.columns and not hf_analysis_df['mid_price_change'].empty:
            price_pressure_zone = hf_analysis_df['mid_price_change'] < 0
            if 'main_force_ofi' in hf_analysis_df.columns and not hf_analysis_df['main_force_ofi'].empty:
                mf_resilience_ofi = hf_analysis_df.loc[price_pressure_zone, 'main_force_ofi'].clip(lower=0).sum()
                total_mf_positive_ofi = hf_analysis_df['main_force_ofi'].clip(lower=0).sum()
                if total_mf_positive_ofi > 0:
                    resilience_component = mf_resilience_ofi / total_mf_positive_ofi
        metrics['main_force_conviction_index'] = (0.4 * aggressiveness_component + 0.4 * cost_tolerance_component + 0.2 * resilience_component) * 100
        mf_trades = hf_features['mf_trades']
        total_mf_vol = hf_features['total_mf_vol']
        if not mf_trades.empty and 'prev_mid_price' in mf_trades.columns:
            buy_trades_mask = mf_trades['type'] == 'B'
            sell_trades_mask = mf_trades['type'] == 'S'
            mf_trades['slippage'] = np.nan
            mf_trades.loc[buy_trades_mask, 'slippage'] = (mf_trades.loc[buy_trades_mask, 'price'] - mf_trades.loc[buy_trades_mask, 'prev_mid_price']).values
            mf_trades.loc[sell_trades_mask, 'slippage'] = (mf_trades.loc[sell_trades_mask, 'prev_mid_price'] - mf_trades.loc[sell_trades_mask, 'price']).values
            mf_trades['slippage'] = mf_trades['slippage'].clip(lower=0)
            if total_mf_vol > 0:
                weighted_avg_slippage = (mf_trades['slippage'] * mf_trades['volume']).sum() / total_mf_vol
                if pd.notna(atr) and atr > 0:
                    metrics['main_force_slippage_index'] = (weighted_avg_slippage / atr) * 100
            if total_mf_vol > 0:
                offensive_volume = hf_features['offensive_volume']
                passive_volume = hf_features['passive_volume']
                metrics['main_force_posture_index'] = ((offensive_volume - passive_volume) / total_mf_vol) * 100
                metrics['main_force_activity_ratio'] = (total_mf_vol / daily_total_volume) * 100 if daily_total_volume > 0 else np.nan
                mf_buy_vol = hf_features['main_force_daily_buy_volume']
                mf_sell_vol = hf_features['main_force_daily_sell_volume']
                mf_total_activity_vol = mf_buy_vol + mf_sell_vol
                if mf_total_activity_vol > 0:
                    mf_net_vol = mf_buy_vol - mf_sell_vol
                    metrics['main_force_flow_directionality'] = (mf_net_vol / mf_total_activity_vol) * 100
        return metrics

    @staticmethod
    def _calculate_ofi_based_metrics(context: dict) -> dict:
        """
        【V72.2 · 主力订单流失衡聚合版 - 空数据鲁棒性增强】
        - 核心重构: `main_force_ofi` 和 `retail_ofi` (以及其买卖分量) 现在基于 `hf_analysis_df` 中
                    新定义的 `main_force_ofi` 和 `retail_ofi` 列（代表实际执行的净主动成交量）进行聚合计算，
                    并归一化为 [-1, 1] 的比率。
                    这确保了日度聚合指标反映的是主力/散户的实际成交行为。
        - 核心修复: `microstructure_efficiency_index` 现在使用 `hf_analysis_df['main_force_ofi']`
                    与 `mid_price_change` 进行相关性计算，以反映执行效率。
        - 核心增强: 增加对 `hf_analysis_df` 是否为空的检查，避免在无高频数据时引发 `KeyError`。
        """
        hf_analysis_df = context['hf_analysis_df']
        metrics = {
            'main_force_ofi': np.nan,
            'retail_ofi': np.nan,
            'main_force_buy_ofi': np.nan,
            'main_force_sell_ofi': np.nan,
            'retail_buy_ofi': np.nan,
            'retail_sell_ofi': np.nan,
            'microstructure_efficiency_index': np.nan,
        }
        if hf_analysis_df.empty:
            return metrics
        # --- 计算主力订单流失衡比率 (基于实际执行成交量) ---
        mf_net_ofi_sum = hf_analysis_df['main_force_ofi'].sum()
        mf_abs_ofi_sum = hf_analysis_df['main_force_ofi'].abs().sum()
        mf_buy_ofi_sum = hf_analysis_df['main_force_ofi'].clip(lower=0).sum()
        mf_sell_ofi_sum = hf_analysis_df['main_force_ofi'].clip(upper=0).abs().sum()
        if mf_abs_ofi_sum > 0:
            metrics['main_force_ofi'] = mf_net_ofi_sum / mf_abs_ofi_sum
            metrics['main_force_buy_ofi'] = mf_buy_ofi_sum / mf_abs_ofi_sum
            metrics['main_force_sell_ofi'] = mf_sell_ofi_sum / mf_abs_ofi_sum
        else:
            metrics['main_force_ofi'] = 0.0
            metrics['main_force_buy_ofi'] = 0.0
            metrics['main_force_sell_ofi'] = 0.0
        # --- 计算散户订单流失衡比率 (基于实际执行成交量) ---
        retail_net_ofi_sum = hf_analysis_df['retail_ofi'].sum()
        retail_abs_ofi_sum = hf_analysis_df['retail_ofi'].abs().sum()
        retail_buy_ofi_sum = hf_analysis_df['retail_ofi'].clip(lower=0).sum()
        retail_sell_ofi_sum = hf_analysis_df['retail_ofi'].clip(upper=0).abs().sum()
        if retail_abs_ofi_sum > 0:
            metrics['retail_ofi'] = retail_net_ofi_sum / retail_abs_ofi_sum
            metrics['retail_buy_ofi'] = retail_buy_ofi_sum / retail_abs_ofi_sum
            metrics['retail_sell_ofi'] = retail_sell_ofi_sum / retail_abs_ofi_sum
        else:
            metrics['retail_ofi'] = 0.0
            metrics['retail_buy_ofi'] = 0.0
            metrics['retail_sell_ofi'] = 0.0
        # --- 更新 microstructure_efficiency_index 使用新的执行订单流 ---
        mf_ofi_series = hf_analysis_df['main_force_ofi']
        price_change_series = hf_analysis_df['mid_price_change']
        if mf_ofi_series.var() > 0 and price_change_series.var() > 0:
            correlation = mf_ofi_series.corr(price_change_series)
            metrics['microstructure_efficiency_index'] = correlation
        return metrics

    @staticmethod
    def _calculate_level5_order_flow_metrics(context: dict) -> dict:
        hf_analysis_df = context['hf_analysis_df']
        common_data = context['common_data']
        metrics = {
            'main_force_level5_ofi': np.nan,
            'main_force_level5_buy_ofi': np.nan,
            'main_force_level5_sell_ofi': np.nan,
            'retail_level5_ofi': np.nan,
            'retail_level5_buy_ofi': np.nan,
            'retail_level5_sell_ofi': np.nan,
            'main_force_level5_ofi_dynamic': np.nan,
            'retail_level5_ofi_dynamic': np.nan,
            'mf_hidden_bid_replenishment_ratio': np.nan,
            'mf_hidden_ask_replenishment_ratio': np.nan,
            'mf_order_cancellation_ratio': np.nan,
        }
        if hf_analysis_df.empty:
            return metrics
        Q_threshold = 1000 * 100
        weights_arr = np.array([1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float64)
        # Prepare data for Numba function
        buy_volumes_cols = [f'buy_volume{i}' for i in range(1, 6)]
        sell_volumes_cols = [f'sell_volume{i}' for i in range(1, 6)]
        buy_prices_cols = [f'buy_price{i}' for i in range(1, 6)]
        sell_prices_cols = [f'sell_price{i}' for i in range(1, 6)]
        # Ensure all required columns exist, fill with 0 if not
        for col in buy_volumes_cols + sell_volumes_cols + buy_prices_cols + sell_prices_cols:
            if col not in hf_analysis_df.columns:
                hf_analysis_df[col] = 0.0
        buy_volumes_arr = hf_analysis_df[buy_volumes_cols].values.astype(np.float64)
        sell_volumes_arr = hf_analysis_df[sell_volumes_cols].values.astype(np.float64)
        buy_prices_arr = hf_analysis_df[buy_prices_cols].values.astype(np.float64)
        sell_prices_arr = hf_analysis_df[sell_prices_cols].values.astype(np.float64)
        (main_force_ofi_snapshots_arr, retail_ofi_snapshots_arr,
         mf_replenishment_bid_events, mf_replenishment_ask_events,
         mf_cancellation_volume, mf_total_posted_volume, num_rows) = \
            _numba_calculate_level5_ofi_components(
                buy_volumes_arr, sell_volumes_arr,
                buy_prices_arr, sell_prices_arr,
                float(Q_threshold), weights_arr
            )
        main_force_ofi_series = pd.Series(main_force_ofi_snapshots_arr, index=hf_analysis_df.index)
        retail_ofi_series = pd.Series(retail_ofi_snapshots_arr, index=hf_analysis_df.index)
        time_diffs = hf_analysis_df.index.to_series().diff().dt.total_seconds().fillna(0)
        total_time = time_diffs.sum()
        if total_time > 0:
            metrics['main_force_level5_ofi'] = np.average(main_force_ofi_series.dropna(), weights=time_diffs[main_force_ofi_series.notna()])
            metrics['main_force_level5_buy_ofi'] = np.average(main_force_ofi_series.clip(lower=0).dropna(), weights=time_diffs[main_force_ofi_series.notna()])
            metrics['main_force_level5_sell_ofi'] = np.average(main_force_ofi_series.clip(upper=0).dropna(), weights=time_diffs[main_force_ofi_series.notna()])
            metrics['retail_level5_ofi'] = np.average(retail_ofi_series.dropna(), weights=time_diffs[retail_ofi_series.notna()])
            metrics['retail_level5_buy_ofi'] = np.average(retail_ofi_series.clip(lower=0).dropna(), weights=time_diffs[retail_ofi_series.notna()])
            metrics['retail_level5_sell_ofi'] = np.average(retail_ofi_series.clip(upper=0).dropna(), weights=time_diffs[retail_ofi_series.notna()])
            metrics['main_force_level5_ofi_dynamic'] = main_force_ofi_series.diff().mean()
            metrics['retail_level5_ofi_dynamic'] = retail_ofi_series.diff().mean()
        else:
            metrics['main_force_level5_ofi'] = main_force_ofi_series.mean()
            metrics['main_force_level5_buy_ofi'] = main_force_ofi_series.clip(lower=0).mean()
            metrics['main_force_level5_sell_ofi'] = main_force_ofi_series.clip(upper=0).mean()
            metrics['retail_level5_ofi'] = retail_ofi_series.mean()
            metrics['retail_level5_buy_ofi'] = retail_ofi_series.clip(lower=0).mean()
            metrics['retail_level5_sell_ofi'] = retail_ofi_series.clip(upper=0).mean()
            metrics['main_force_level5_ofi_dynamic'] = main_force_ofi_series.diff().mean()
            metrics['retail_level5_ofi_dynamic'] = retail_ofi_series.diff().mean()
        total_replenishment_events = mf_replenishment_bid_events + mf_replenishment_ask_events
        if total_replenishment_events > 0:
            metrics['mf_hidden_bid_replenishment_ratio'] = mf_replenishment_bid_events / total_replenishment_events
            metrics['mf_hidden_ask_replenishment_ratio'] = mf_replenishment_ask_events / total_replenishment_events
        if mf_total_posted_volume > 0:
            metrics['mf_order_cancellation_ratio'] = mf_cancellation_volume / mf_total_posted_volume
        return metrics

    @staticmethod
    def _calculate_main_force_intraday_intent(context: dict) -> dict:
        """
        【V1.0 · 主力日内意图流】
        - 核心职责: 融合高频tick和Level5数据中提取的主力行为特征，生成日度的“主力日内意图流”指标。
        - 核心逻辑: 综合主力主动买卖意图、订单簿控制意图、承接/派发意图，并结合日度校准资金流。
        参数:
            context (dict): 包含所有计算所需数据和特征的上下文字典。
        返回:
            dict: 包含 'main_force_intraday_intent_D' 指标的字典。
        """
        metrics = {'main_force_intraday_intent_D': np.nan}
        daily_data = context['daily_data']
        hf_features = context['hf_features']
        mf_metrics = context['mf_metrics'] # 包含 _calculate_level5_order_flow_metrics 的结果
        atr = daily_data.get('atr_14d', np.nan)
        if pd.isna(atr) or atr <= 0:
            return metrics
        # 1. 主力主动攻击意图 (Aggressive Intent)
        # 衡量主力主动推高或打压价格的意愿
        # 使用主力主动买卖量和平均价格冲击
        main_force_aggressive_buy_volume = hf_features.get('main_force_aggressive_buy_volume', 0.0)
        main_force_aggressive_sell_volume = hf_features.get('main_force_aggressive_sell_volume', 0.0)
        main_force_avg_price_impact = hf_features.get('main_force_avg_price_impact', np.nan)
        # 归一化价格冲击，使其在ATR范围内
        normalized_price_impact = (main_force_avg_price_impact / atr) if pd.notna(main_force_avg_price_impact) else 0.0
        # 简单归一化主动买卖量，可以考虑用日总成交量或主力总成交量归一化
        total_aggressive_volume = main_force_aggressive_buy_volume + main_force_aggressive_sell_volume
        if total_aggressive_volume > 0:
            aggressive_buy_ratio = main_force_aggressive_buy_volume / total_aggressive_volume
            aggressive_sell_ratio = main_force_aggressive_sell_volume / total_aggressive_volume
            net_aggressive_intent = (aggressive_buy_ratio - aggressive_sell_ratio) * (1 + normalized_price_impact)
        else:
            net_aggressive_intent = 0.0
        # 2. 主力订单簿控制意图 (Order Book Control Intent)
        # 衡量主力通过挂单对市场预期的影响
        main_force_level5_ofi = mf_metrics.get('main_force_level5_ofi', 0.0)
        main_force_level5_ofi_dynamic = mf_metrics.get('main_force_level5_ofi_dynamic', 0.0)
        mf_hidden_bid_replenishment_ratio = mf_metrics.get('mf_hidden_bid_replenishment_ratio', 0.0)
        mf_hidden_ask_replenishment_ratio = mf_metrics.get('mf_hidden_ask_replenishment_ratio', 0.0)
        mf_order_cancellation_ratio = mf_metrics.get('mf_order_cancellation_ratio', 0.0)
        # 融合Level5指标，隐藏订单补充为正向，撤单为负向
        order_book_control_score = (
            0.5 * main_force_level5_ofi +
            0.2 * main_force_level5_ofi_dynamic +
            0.2 * (mf_hidden_bid_replenishment_ratio - mf_hidden_ask_replenishment_ratio) -
            0.1 * mf_order_cancellation_ratio # 撤单可能暗示虚假挂单或意图改变
        )
        # 3. 主力承接/派发意图 (Absorption/Distribution Intent)
        # 衡量主力在价格波动中是吸筹还是派发
        main_force_passive_buy_volume = hf_features.get('main_force_passive_buy_volume', 0.0)
        main_force_passive_sell_volume = hf_features.get('main_force_passive_sell_volume', 0.0)
        total_passive_volume = main_force_passive_buy_volume + main_force_passive_sell_volume
        if total_passive_volume > 0:
            net_passive_intent = (main_force_passive_buy_volume - main_force_passive_sell_volume) / total_passive_volume
        else:
            net_passive_intent = 0.0
        # 4. 日度校准资金流 (作为基础参考)
        main_force_net_flow_calibrated = context['main_force_net_flow_calibrated']
        # 融合所有组件，权重可调
        # 假设我们更看重主动攻击和订单簿控制，被动行为次之，日度净流作为背景
        main_force_intraday_intent = (
            0.4 * net_aggressive_intent +
            0.3 * order_book_control_score +
            0.2 * net_passive_intent +
            0.1 * (main_force_net_flow_calibrated / abs(main_force_net_flow_calibrated) if main_force_net_flow_calibrated != 0 else 0.0) # 归一化为方向
        )
        # 最终归一化到 [-1, 1]
        metrics['main_force_intraday_intent_D'] = np.tanh(main_force_intraday_intent) # 使用tanh进行平滑归一化
        return metrics

    @staticmethod
    def _calculate_order_book_metrics(context: dict) -> dict:
        """
        【V71.0 · 终极生产版 - 空数据鲁棒性增强】
        【V72.0 · 资金流拆分版】
        - 核心增强: 拆分 `order_book_clearing_rate` 和 `order_book_imbalance` 为买卖双方贡献。
        - 核心增强: 增加对 `hf_analysis_df` 是否为空的检查，避免在无高频数据时引发 `KeyError`。
        - 核心修复: 在访问 `imbalance` 和 `liquidity_supply_ratio` 之前，增加列存在性检查。
        """
        hf_analysis_df = context['hf_analysis_df']
        common_data = context['common_data']
        import numpy as np
        metrics = {}
        if hf_analysis_df.empty:
            return metrics
        daily_total_volume = common_data['daily_total_volume']
        large_orders_df = hf_analysis_df[hf_analysis_df['amount'] > 200000]
        if not large_orders_df.empty:
            metrics['observed_large_order_size_avg'] = large_orders_df['amount'].mean()
        up_ticks = hf_analysis_df[hf_analysis_df['mid_price_change'] > 0]
        down_ticks = hf_analysis_df[hf_analysis_df['mid_price_change'] < 0]
        if not up_ticks.empty and not down_ticks.empty and up_ticks['mid_price_change'].sum() > 0 and down_ticks['mid_price_change'].abs().sum() > 0:
            vol_per_tick_up = up_ticks['volume'].sum() / (up_ticks['mid_price_change'].sum() * 100)
            vol_per_tick_down = down_ticks['volume'].sum() / (down_ticks['mid_price_change'].abs().sum() * 100)
            if vol_per_tick_down > 1e-9:
                asymmetry_ratio = vol_per_tick_up / vol_per_tick_down
                metrics['micro_price_impact_asymmetry'] = np.log(asymmetry_ratio) if asymmetry_ratio > 1e-9 else np.nan
        if 'prev_a1_p' in hf_analysis_df.columns and 'prev_b1_p' in hf_analysis_df.columns:
            ask_clearing_mask = (hf_analysis_df['type'] == 'B') & (hf_analysis_df['price'] == hf_analysis_df['prev_a1_p'])
            ask_clearing_vol = hf_analysis_df.loc[ask_clearing_mask, 'volume'].sum()
            bid_clearing_mask = (hf_analysis_df['type'] == 'S') & (hf_analysis_df['price'] == hf_analysis_df['prev_b1_p'])
            bid_clearing_vol = hf_analysis_df.loc[bid_clearing_mask, 'volume'].sum()
            total_cleared_vol = ask_clearing_vol + bid_clearing_vol
            if daily_total_volume > 0:
                metrics['order_book_clearing_rate'] = (total_cleared_vol / daily_total_volume) * 100
                metrics['buy_order_book_clearing_rate'] = (ask_clearing_vol / daily_total_volume) * 100
                metrics['sell_order_book_clearing_rate'] = (bid_clearing_vol / daily_total_volume) * 100
        try:
            time_diffs = hf_analysis_df.index.to_series().diff().dt.total_seconds().fillna(0)
            if time_diffs.sum() > 0:
                if 'imbalance' in hf_analysis_df.columns: # 增加列存在性检查
                    metrics['order_book_imbalance'] = np.average(hf_analysis_df['imbalance'].dropna(), weights=time_diffs[hf_analysis_df['imbalance'].notna()]) * 100
                if 'liquidity_supply_ratio' in hf_analysis_df.columns: # 增加列存在性检查
                    metrics['order_book_liquidity_supply'] = np.average(hf_analysis_df['liquidity_supply_ratio'].dropna(), weights=time_diffs[hf_analysis_df['liquidity_supply_ratio'].notna()])
                bid_liquidity_cols = [f'buy_volume{i}' for i in range(1, 6)]
                ask_liquidity_cols = [f'sell_volume{i}' for i in range(1, 6)]
                # 检查所有 Level 5 订单簿列是否存在
                if all(col in hf_analysis_df.columns for col in bid_liquidity_cols) and all(col in hf_analysis_df.columns for col in ask_liquidity_cols):
                    bid_depth_series = hf_analysis_df[bid_liquidity_cols].sum(axis=1)
                    ask_depth_series = hf_analysis_df[ask_liquidity_cols].sum(axis=1)
                    metrics['bid_side_liquidity'] = np.average(bid_depth_series.dropna(), weights=time_diffs[bid_depth_series.notna()]) if bid_depth_series.notna().any() else np.nan
                    metrics['ask_side_liquidity'] = np.average(ask_depth_series.dropna(), weights=time_diffs[ask_depth_series.notna()]) if ask_depth_series.notna().any() else np.nan
                if 'market_vol_delta' in hf_analysis_df.columns and 'imbalance' in hf_analysis_df.columns and hf_analysis_df['imbalance'].var() > 1e-9 and hf_analysis_df['market_vol_delta'].var() > 1e-9:
                    correlation_value = hf_analysis_df['imbalance'].corr(hf_analysis_df['market_vol_delta'])
                    metrics['imbalance_effectiveness'] = correlation_value
        except Exception:
            pass
        try:
            df_static = hf_analysis_df.copy()
            large_order_threshold_value = 500000
            # 检查 Level 5 订单簿列是否存在
            required_level5_cols_for_pressure = ['sell_volume1', 'sell_price1', 'sell_volume2', 'sell_price2', 'buy_volume1', 'buy_price1', 'buy_volume2', 'buy_price2']
            if all(col in df_static.columns for col in required_level5_cols_for_pressure):
                pressure_mask = (df_static['sell_volume1'] * df_static['sell_price1'] > large_order_threshold_value) | (df_static['sell_volume2'] * df_static['sell_price2'] > large_order_threshold_value)
                support_mask = (df_static['buy_volume1'] * df_static['buy_price1'] > large_order_threshold_value) | (df_static['buy_volume2'] * df_static['buy_price2'] > large_order_threshold_value)
                time_diffs = df_static.index.to_series().diff().dt.total_seconds().fillna(0)
                pressure_strength = 0; support_strength = 0
                if 'market_vol_delta' in df_static.columns:
                    market_activity = df_static['market_vol_delta'].rolling(window=20, min_periods=1).mean().replace(0, np.nan)
                    activity_factor = 1 / np.log1p(market_activity)
                    pressure_strength = (time_diffs * activity_factor)[pressure_mask].sum()
                    support_strength = (time_diffs * activity_factor)[support_mask].sum()
                else:
                    pressure_strength = time_diffs[pressure_mask].sum(); support_strength = time_diffs[support_mask].sum()
                total_trading_seconds = (df_static.index.max() - df_static.index.min()).total_seconds()
                if total_trading_seconds > 0:
                    metrics['large_order_pressure'] = (pressure_strength / total_trading_seconds) * 100
                    metrics['large_order_support'] = (support_strength / total_trading_seconds) * 100
            else:
                metrics['large_order_pressure'] = np.nan; metrics['large_order_support'] = np.nan
        except Exception:
            metrics['large_order_pressure'] = np.nan; metrics['large_order_support'] = np.nan
        if 'prev_a1_p' in hf_analysis_df.columns and 'prev_b1_p' in hf_analysis_df.columns and 'prev_a1_v' in hf_analysis_df.columns and 'prev_b1_v' in hf_analysis_df.columns:
            try:
                buy_exhaustion_mask = hf_analysis_df['sell_price1'] > hf_analysis_df['prev_a1_p']
                buy_exhausted_vol = hf_analysis_df.loc[buy_exhaustion_mask, 'prev_a1_v'].sum()
                sell_exhaustion_mask = hf_analysis_df['buy_price1'] < hf_analysis_df['prev_b1_p']
                sell_exhausted_vol = hf_analysis_df.loc[sell_exhaustion_mask, 'prev_b1_v'].sum()
                if daily_total_volume > 0:
                    metrics['buy_quote_exhaustion_rate'] = (buy_exhausted_vol / daily_total_volume) * 100
                    metrics['sell_quote_exhaustion_rate'] = (sell_exhausted_vol / daily_total_volume) * 100
            except Exception:
                metrics['buy_quote_exhaustion_rate'] = np.nan; metrics['sell_quote_exhaustion_rate'] = np.nan
        return metrics

    @staticmethod
    def _calculate_opening_battle_metrics(context: dict) -> dict:
        """
        【优化版】使用 between_time 替代 index.time 比较，提升时间切片效率。
        """
        intraday_data = context['intraday_data']
        hf_analysis_df = context['hf_analysis_df']
        common_data = context['common_data']
        import numpy as np
        metrics = {}
        atr = common_data['atr']
        # 优化点：使用 between_time 进行快速切片
        opening_battle_df = intraday_data.between_time('09:30', '09:45')
        if not opening_battle_df.empty and len(opening_battle_df) > 1 and pd.notna(atr) and atr > 0:
            if not hf_analysis_df.empty:
                # 优化点：同理优化高频数据切片
                opening_hf_df = hf_analysis_df.between_time('09:30', '09:45')
                if not opening_hf_df.empty:
                    price_gain_hf = (opening_hf_df['price'].iloc[-1] - opening_hf_df['price'].iloc[0]) / atr
                    mf_ofi_opening = opening_hf_df['main_force_ofi'].sum()
                    total_abs_ofi_opening = opening_hf_df['ofi'].abs().sum()
                    mf_ofi_dominance = mf_ofi_opening / total_abs_ofi_opening if total_abs_ofi_opening > 0 else 0
                    metrics['opening_battle_result'] = price_gain_hf * (1 + mf_ofi_dominance) * 100
                    mf_buy_ofi_opening = opening_hf_df['main_force_ofi'].clip(lower=0).sum()
                    mf_sell_ofi_opening = opening_hf_df['main_force_ofi'].clip(upper=0).sum()
                    metrics['opening_buy_strength'] = (mf_buy_ofi_opening / total_abs_ofi_opening) * 100 if total_abs_ofi_opening > 0 else np.nan
                    metrics['opening_sell_strength'] = (abs(mf_sell_ofi_opening) / total_abs_ofi_opening) * 100 if total_abs_ofi_opening > 0 else np.nan
            else:
                if 'close' in opening_battle_df.columns and 'open' in opening_battle_df.columns and 'vol_shares' in opening_battle_df.columns and 'minute_vwap' in opening_battle_df.columns and 'main_force_net_vol' in opening_battle_df.columns:
                    price_gain = (opening_battle_df['close'].iloc[-1] - opening_battle_df['open'].iloc[0]) / atr
                    battle_amount = (opening_battle_df['vol_shares'] * opening_battle_df['minute_vwap']).sum()
                    if battle_amount > 0:
                        mf_power = opening_battle_df['main_force_net_vol'].sum() * opening_battle_df['minute_vwap'].mean() / battle_amount
                        metrics['opening_battle_result'] = np.sign(price_gain) * np.sqrt(abs(price_gain)) * (1 + mf_power) * 100
                        mf_buy_vol_opening = opening_battle_df['main_force_buy_vol'].sum()
                        mf_sell_vol_opening = opening_battle_df['main_force_sell_vol'].sum()
                        total_vol_opening = opening_battle_df['vol_shares'].sum()
                        metrics['opening_buy_strength'] = (mf_buy_vol_opening / total_vol_opening) * 100 if total_vol_opening > 0 else np.nan
                        metrics['opening_sell_strength'] = (mf_sell_vol_opening / total_vol_opening) * 100 if total_vol_opening > 0 else np.nan
        return metrics

    @staticmethod
    def _calculate_shadow_metrics(context: dict) -> dict:
        """
        【优化版】直接在 mf_trades 上进行价格过滤，移除低效的 index.intersection 操作。
        """
        hf_analysis_df = context['hf_analysis_df']
        common_data = context['common_data']
        hf_features = context['hf_features']
        import numpy as np
        metrics = {}
        if hf_analysis_df.empty:
            return metrics
        day_open, day_close = common_data['day_open'], common_data['day_close']
        day_high, day_low = common_data['day_high'], common_data['day_low']
        atr = common_data.get('atr', 0)
        daily_total_amount = common_data.get('daily_total_amount', 0)
        mf_trades = hf_features.get('mf_trades', pd.DataFrame())
        if pd.notna(day_open) and pd.notna(day_close) and pd.notna(day_high) and pd.notna(day_low) and pd.notna(atr) and atr > 0 and daily_total_amount > 0 and not mf_trades.empty:
            day_range = day_high - day_low
            if day_range <= 0:
                return metrics
            market_value_efficiency = (day_range / atr) / (daily_total_amount / 10000)
            body_high, body_low = max(day_open, day_close), min(day_open, day_close)
            # 下影线逻辑（主力吸筹）
            if day_low < body_low:
                price_recovery_norm = (body_low - day_low) / atr
                # --- 优化点：直接在 mf_trades 上筛选价格，替代 intersection ---
                mf_trades_in_shadow = mf_trades[mf_trades['price'] < body_low]
                # --- 结束优化 ---
                if not mf_trades_in_shadow.empty:
                    mf_buy_amount = mf_trades_in_shadow[mf_trades_in_shadow['type'] == 'B']['amount'].sum()
                    mf_sell_amount = mf_trades_in_shadow[mf_trades_in_shadow['type'] == 'S']['amount'].sum()
                    mf_net_buy_amount_10k = (mf_buy_amount - mf_sell_amount) / 10000
                    if mf_net_buy_amount_10k > 0 and market_value_efficiency > 0:
                        absorption_efficiency = price_recovery_norm / mf_net_buy_amount_10k
                        normalized_strength = absorption_efficiency / market_value_efficiency
                        compressed_strength = np.log1p(normalized_strength)
                        metrics['lower_shadow_absorption_strength'] = np.tanh(compressed_strength) * 100
            # 上影线逻辑（主力派发）
            if day_high > body_high:
                price_rejection_norm = (day_high - body_high) / atr
                # --- 优化点：直接在 mf_trades 上筛选价格 ---
                mf_trades_in_shadow = mf_trades[mf_trades['price'] > body_high]
                # --- 结束优化 ---
                if not mf_trades_in_shadow.empty:
                    mf_buy_amount = mf_trades_in_shadow[mf_trades_in_shadow['type'] == 'B']['amount'].sum()
                    mf_sell_amount = mf_trades_in_shadow[mf_trades_in_shadow['type'] == 'S']['amount'].sum()
                    mf_net_sell_amount_10k = (mf_sell_amount - mf_buy_amount) / 10000
                    if mf_net_sell_amount_10k > 0 and market_value_efficiency > 0:
                        rejection_efficiency = price_rejection_norm / mf_net_sell_amount_10k
                        normalized_pressure = rejection_efficiency / market_value_efficiency
                        compressed_pressure = np.log1p(normalized_pressure)
                        metrics['upper_shadow_selling_pressure'] = np.tanh(compressed_pressure) * 100
        return metrics

    @staticmethod
    def _calculate_dip_rally_metrics(context: dict, raw_hf_df: pd.DataFrame) -> dict:
        """
        【优化版】使用 searchsorted 优化 DataFrame 切片性能。
        """
        intraday_data = context['intraday_data']
        hf_analysis_df = context['hf_analysis_df']
        common_data = context['common_data']
        from scipy.signal import find_peaks
        from datetime import time
        import numpy as np
        import pandas as pd
        metrics = {}
        if hf_analysis_df.empty or intraday_data.empty or raw_hf_df.empty:
            return metrics
        daily_vwap = common_data['daily_vwap']
        atr = common_data['atr']
        daily_total_amount = common_data.get('daily_total_amount', 0)
        day_high, day_low = common_data['day_high'], common_data['day_low']
        day_range = day_high - day_low if day_high > day_low else atr * 0.1
        # 1. 精细化下跌吸收指标计算
        if pd.notna(daily_vwap) and pd.notna(atr) and atr > 0:
            dip_mask = raw_hf_df['price'] < daily_vwap
            dip_hf_df = raw_hf_df[dip_mask].copy()
            if not dip_hf_df.empty:
                volume_threshold = dip_hf_df['volume'].quantile(0.8)
                mf_trades_in_dip = dip_hf_df[dip_hf_df['volume'] >= volume_threshold]
                retail_trades_in_dip = dip_hf_df[dip_hf_df['volume'] < volume_threshold]
                mf_aggressive_buy = mf_trades_in_dip[mf_trades_in_dip['type'] == 'B']['volume'].sum()
                mf_aggressive_sell = mf_trades_in_dip[mf_trades_in_dip['type'] == 'S']['volume'].sum()
                mf_net_aggressive = mf_aggressive_buy - mf_aggressive_sell
                retail_aggressive_sell = retail_trades_in_dip[retail_trades_in_dip['type'] == 'S']['volume'].sum()
                retail_aggressive_buy = retail_trades_in_dip[retail_trades_in_dip['type'] == 'B']['volume'].sum()
                retail_net_aggressive_sell = retail_aggressive_sell - retail_aggressive_buy
                dip_hf_df['bid_depth'] = dip_hf_df[['buy_volume1', 'buy_volume2', 'buy_volume3', 'buy_volume4', 'buy_volume5']].sum(axis=1)
                dip_hf_df['ask_depth'] = dip_hf_df[['sell_volume1', 'sell_volume2', 'sell_volume3', 'sell_volume4', 'sell_volume5']].sum(axis=1)
                avg_bid_ask_ratio = (dip_hf_df['bid_depth'] / (dip_hf_df['ask_depth'] + 1e-10)).mean()
                price_depth = (daily_vwap - dip_hf_df['price']).clip(lower=0)
                total_price_weighted_volume = (price_depth * dip_hf_df['volume']).sum()
                if total_price_weighted_volume > 0:
                    absorption_efficiency = mf_net_aggressive / total_price_weighted_volume if mf_net_aggressive > 0 else 0
                    retail_pressure_ratio = retail_net_aggressive_sell / dip_hf_df['volume'].sum() if dip_hf_df['volume'].sum() > 0 else 0
                    bid_ask_support = avg_bid_ask_ratio if avg_bid_ask_ratio > 0 else 0.5
                    composite_absorption = absorption_efficiency * max(0, 1 - retail_pressure_ratio) * bid_ask_support * atr
                    metrics['dip_absorption_power'] = np.tanh(composite_absorption) * 100
                    if mf_aggressive_buy > 0:
                        mf_buy_concentration = (mf_trades_in_dip[mf_trades_in_dip['type'] == 'B']['volume'].max() / mf_aggressive_buy) if mf_aggressive_buy > 0 else 0
                        concentration_factor = 1 + mf_buy_concentration
                        absorption_strength = absorption_efficiency * concentration_factor * bid_ask_support * atr
                        metrics['dip_buy_absorption_strength'] = np.tanh(absorption_strength) * 100
                    if retail_net_aggressive_sell > 0 and mf_net_aggressive > 0:
                        resistance_ratio = retail_net_aggressive_sell / mf_net_aggressive
                        ask_depth_change = dip_hf_df['ask_depth'].pct_change().mean() if len(dip_hf_df) > 1 else 0
                        ask_reduction_factor = max(0, -ask_depth_change)
                        resistance_strength = resistance_ratio * (1 + ask_reduction_factor) * bid_ask_support * atr
                        metrics['dip_sell_pressure_resistance'] = np.tanh(resistance_strength) * 100
        # 2. 精细化反弹分布指标计算
        continuous_trading_df = intraday_data[intraday_data.index.time < time(14, 57)].copy()
        if not continuous_trading_df.empty and 'minute_vwap' in continuous_trading_df.columns:
            peaks, peak_properties = find_peaks(continuous_trading_df['minute_vwap'].values, prominence=0.002*atr, width=3)
            troughs, trough_properties = find_peaks(-continuous_trading_df['minute_vwap'].values, prominence=0.002*atr, width=3)
            turning_points = sorted(list(set(np.concatenate(([0], troughs, peaks, [len(continuous_trading_df)-1])))))
            total_mf_net_sell_amount_in_rallies = 0
            total_mf_net_buy_amount_in_rallies = 0
            total_rally_price_change_norm = 0
            total_rally_volume = 0
            rally_distribution_pressures = []
            rally_buy_weakness_ratios = []
            # 确保索引为DatetimeIndex以使用searchsorted
            hf_index = raw_hf_df.index
            for i in range(len(turning_points) - 1):
                start_idx, end_idx = turning_points[i], turning_points[i+1]
                window_df = continuous_trading_df.iloc[start_idx:end_idx+1]
                if window_df.empty or len(window_df) < 2: continue
                vwap_start, vwap_end = window_df['minute_vwap'].iloc[0], window_df['minute_vwap'].iloc[-1]
                price_change = vwap_end - vwap_start
                if price_change > 0 and price_change/vwap_start > 0.001:
                    start_time, end_time = window_df.index[0], window_df.index[-1]
                    # --- 优化：使用 searchsorted 快速定位切片 ---
                    start_loc = hf_index.searchsorted(start_time)
                    end_loc = hf_index.searchsorted(end_time, side='right')
                    rally_hf_df = raw_hf_df.iloc[start_loc:end_loc].copy()
                    if rally_hf_df.empty: continue
                    # --- 结束优化 ---
                    rally_volume = rally_hf_df['volume'].sum()
                    total_rally_volume += rally_volume
                    volume_threshold_rally = rally_hf_df['volume'].quantile(0.8)
                    mf_trades_in_rally = rally_hf_df[rally_hf_df['volume'] >= volume_threshold_rally]
                    mf_aggressive_sell_amt = mf_trades_in_rally[mf_trades_in_rally['type'] == 'S']['amount'].sum()
                    mf_aggressive_buy_amt = mf_trades_in_rally[mf_trades_in_rally['type'] == 'B']['amount'].sum()
                    mf_net_aggressive_sell = mf_aggressive_sell_amt - mf_aggressive_buy_amt
                    total_mf_net_sell_amount_in_rallies += mf_net_aggressive_sell
                    retail_trades_in_rally = rally_hf_df[rally_hf_df['volume'] < volume_threshold_rally]
                    retail_aggressive_buy_amt = retail_trades_in_rally[retail_trades_in_rally['type'] == 'B']['amount'].sum()
                    retail_aggressive_sell_amt = retail_trades_in_rally[retail_trades_in_rally['type'] == 'S']['amount'].sum()
                    retail_net_aggressive_buy = retail_aggressive_buy_amt - retail_aggressive_sell_amt
                    total_mf_net_buy_amount_in_rallies += retail_net_aggressive_buy
                    rally_hf_df['ask_depth'] = rally_hf_df[['sell_volume1', 'sell_volume2', 'sell_volume3', 'sell_volume4', 'sell_volume5']].sum(axis=1)
                    ask_depth_change_rally = rally_hf_df['ask_depth'].pct_change().mean() if len(rally_hf_df) > 1 else 0
                    price_change_norm = price_change / atr if atr > 0 else price_change / (vwap_start * 0.01)
                    total_rally_price_change_norm += price_change_norm
                    momentum_decay = 1.0
                    if len(rally_hf_df) >= 10:
                        first_third = len(rally_hf_df) // 3
                        last_third = len(rally_hf_df) - first_third
                        first_half_price_change = rally_hf_df.iloc[first_third]['price'] - rally_hf_df.iloc[0]['price']
                        second_half_price_change = rally_hf_df.iloc[-1]['price'] - rally_hf_df.iloc[last_third]['price']
                        momentum_decay = second_half_price_change / first_half_price_change if first_half_price_change > 0 else 1.0
                    if price_change_norm > 0 and rally_volume > 0:
                        ask_pressure_factor = 1 + max(0, ask_depth_change_rally)
                        distribution_pressure = (mf_net_aggressive_sell / 10000) * ask_pressure_factor * momentum_decay / price_change_norm
                        rally_distribution_pressures.append(distribution_pressure)
                    if mf_net_aggressive_sell > 0 and retail_net_aggressive_buy > 0:
                        weakness_ratio = retail_net_aggressive_buy / mf_net_aggressive_sell
                        rally_buy_weakness_ratios.append(weakness_ratio)
            if total_rally_price_change_norm > 0 and day_range > 0 and daily_total_amount > 0:
                market_price_cost = (daily_total_amount / 10000) / (day_range / atr) if (day_range / atr) > 0 else 1.0
                if rally_distribution_pressures:
                    avg_rally_pressure = np.mean(rally_distribution_pressures)
                    volume_weight = total_rally_volume / daily_total_amount if daily_total_amount > 0 else 1.0
                    normalized_pressure = avg_rally_pressure * volume_weight / market_price_cost if market_price_cost > 0 else avg_rally_pressure
                    metrics['rally_distribution_pressure'] = np.tanh(normalized_pressure) * 100
                    pressure_std = np.std(rally_distribution_pressures) if len(rally_distribution_pressures) > 1 else 0
                    pressure_concentration = 1.0 / (1.0 + pressure_std) if pressure_std > 0 else 1.0
                    distribution_intensity = normalized_pressure * pressure_concentration
                    metrics['rally_sell_distribution_intensity'] = np.tanh(distribution_intensity) * 100
                if rally_buy_weakness_ratios:
                    avg_weakness_ratio = np.mean(rally_buy_weakness_ratios)
                    weakness_coeff = avg_weakness_ratio * (total_mf_net_sell_amount_in_rallies / 10000) / total_rally_price_change_norm
                    volume_adjusted_weakness = weakness_coeff * (total_rally_volume / daily_total_amount) if daily_total_amount > 0 else weakness_coeff
                    normalized_weakness = volume_adjusted_weakness / market_price_cost if market_price_cost > 0 else volume_adjusted_weakness
                    metrics['rally_buy_support_weakness'] = np.tanh(normalized_weakness) * 100
        return metrics

    @staticmethod
    def _calculate_reversal_metrics(context: dict) -> dict:
        """
        【V71.0 · 终极生产版】(生产环境清洁版)
        """
        intraday_data = context['intraday_data']
        hf_analysis_df = context['hf_analysis_df']
        common_data = context['common_data']
        import numpy as np
        metrics = {}
        day_open, day_close = common_data['day_open'], common_data['day_close']
        day_high, day_low = common_data['day_high'], common_data['day_low']
        atr = common_data['atr']
        daily_total_volume = common_data['daily_total_volume']
        if len(intraday_data) >= 10 and pd.notna(day_open) and pd.notna(day_close) and pd.notna(day_high) and pd.notna(day_low) and pd.notna(atr) and atr > 0 and daily_total_volume > 0:
            day_range = day_high - day_low
            if day_range > 0:
                is_v_shape = (day_close - day_open) > 0
                turn_point_idx = np.argmin(intraday_data['low'].values) if is_v_shape else np.argmax(intraday_data['high'].values)
                if 0 < turn_point_idx < len(intraday_data) - 1:
                    if not hf_analysis_df.empty:
                        turn_point_time = intraday_data.index[turn_point_idx]
                        reversal_phase_hf = hf_analysis_df[hf_analysis_df.index >= turn_point_time]
                        if not reversal_phase_hf.empty:
                            turn_point_price = intraday_data.iloc[turn_point_idx]['low'] if is_v_shape else intraday_data.iloc[turn_point_idx]['high']
                            PriceRecovery_Component = abs(day_close - turn_point_price) / day_range
                            reversal_ofi = reversal_phase_hf['main_force_ofi']
                            CounterAttack_Component = np.tanh(reversal_ofi.sum() / daily_total_volume)
                            power_score = (0.6 * PriceRecovery_Component + 0.4 * CounterAttack_Component)
                            metrics['reversal_power_index'] = power_score * 100
                    else:
                        initial_phase = intraday_data.iloc[:turn_point_idx]
                        reversal_phase = intraday_data.iloc[turn_point_idx:]
                        vol_initial, vol_reversal = initial_phase['vol_shares'].sum(), reversal_phase['vol_shares'].sum()
                        if vol_initial > 0 and vol_reversal > 0 and 'main_force_net_vol' in reversal_phase.columns:
                            turn_point_vwap = intraday_data['minute_vwap'].iloc[turn_point_idx]
                            price_recovery = abs(day_close - turn_point_vwap) / day_range
                            vol_shift = np.log1p(vol_reversal / vol_initial)
                            reversal_mf_net_vol = reversal_phase['main_force_net_vol'].sum()
                            reversal_conviction = reversal_mf_net_vol / vol_reversal if vol_reversal > 0 else 0
                            power_score = price_recovery * vol_shift * reversal_conviction
                            metrics['reversal_power_index'] = power_score if is_v_shape else -power_score
        return metrics

    @staticmethod
    def _calculate_closing_metrics(context: dict) -> dict:
        intraday_data = context['intraday_data']
        hf_analysis_df = context['hf_analysis_df']
        common_data = context['common_data']
        import numpy as np
        metrics = {}
        if hf_analysis_df.empty:
            return metrics
        day_close = common_data['day_close']
        daily_vwap = common_data['daily_vwap']
        atr = common_data['atr']
        continuous_trading_df = intraday_data.between_time('00:00', '14:57', inclusive='left').copy()
        if not continuous_trading_df.empty and pd.notna(atr) and atr > 0:
            auction_df = intraday_data.between_time('14:57', '15:00', inclusive='both')
            if not auction_df.empty:
                avg_minute_vol = continuous_trading_df['vol_shares'].mean()
                auction_vol = auction_df['vol_shares'].sum()
                VolumeAnomaly = np.log1p((auction_vol / 3) / avg_minute_vol) if avg_minute_vol > 0 else 0.0
                pre_auction_df = hf_analysis_df.between_time('00:00', '14:57', inclusive='left')
                if not pre_auction_df.empty:
                    pre_auction_snapshot = pre_auction_df.iloc[-1]
                    pre_auction_mid = pre_auction_snapshot['mid_price']
                    pre_auction_imbalance = pre_auction_snapshot['imbalance'] if 'imbalance' in pre_auction_snapshot else np.nan
                    PriceDeviation = (day_close - pre_auction_mid) / atr if pd.notna(pre_auction_mid) else 0.0
                    Deception = -np.sign(PriceDeviation) * pre_auction_imbalance if pd.notna(pre_auction_imbalance) else 0.0
                    metrics['closing_auction_ambush'] = PriceDeviation * VolumeAnomaly * (1 + Deception) * 100
                    auction_hf_df = hf_analysis_df.between_time('14:57', '15:00')
                    large_auction_trades = auction_hf_df[auction_hf_df['amount'] > 200000]
                    mf_auction_buy_vol = large_auction_trades[large_auction_trades['type'] == 'B']['volume'].sum()
                    mf_auction_sell_vol = large_auction_trades[large_auction_trades['type'] == 'S']['volume'].sum()
                    total_auction_vol = auction_hf_df['volume'].sum()
                    if total_auction_vol > 0:
                        metrics['closing_auction_buy_ambush'] = (mf_auction_buy_vol / total_auction_vol) * PriceDeviation * VolumeAnomaly * 100
                        metrics['closing_auction_sell_ambush'] = (mf_auction_sell_vol / total_auction_vol) * PriceDeviation * VolumeAnomaly * 100
                else:
                    pre_auction_close = continuous_trading_df['close'].iloc[-1]
                    PriceImpact = (day_close - pre_auction_close) / atr if pd.notna(pre_auction_close) else 0.0
                    metrics['closing_auction_ambush'] = PriceImpact * VolumeAnomaly * 100
                    mf_auction_buy_vol_fallback = auction_df['main_force_buy_vol'].sum()
                    mf_auction_sell_vol_fallback = auction_df['main_force_sell_vol'].sum()
                    total_auction_vol_fallback = auction_df['vol_shares'].sum()
                    if total_auction_vol_fallback > 0:
                        metrics['closing_auction_buy_ambush'] = (mf_auction_buy_vol_fallback / total_auction_vol_fallback) * PriceImpact * VolumeAnomaly * 100
                        metrics['closing_auction_sell_ambush'] = (mf_auction_sell_vol_fallback / total_auction_vol_fallback) * PriceImpact * VolumeAnomaly * 100
            posturing_df = continuous_trading_df.between_time('14:30', '15:00')
            if pd.notna(daily_vwap) and not posturing_df.empty:
                posturing_hf_df = hf_analysis_df.between_time('14:30', '15:00')
                if not posturing_hf_df.empty:
                    time_diffs = posturing_hf_df.index.to_series().diff().dt.total_seconds().fillna(0)
                    if time_diffs.sum() > 0:
                        avg_imbalance = np.nan
                        if 'imbalance' in posturing_hf_df.columns:
                            avg_imbalance = np.average(posturing_hf_df['imbalance'].dropna(), weights=time_diffs[posturing_hf_df['imbalance'].notna()])
                        avg_spread = (posturing_hf_df['sell_price1'] - posturing_hf_df['buy_price1']).mean()
                        normalized_imbalance = avg_imbalance * (avg_spread / atr) if pd.notna(avg_imbalance) and pd.notna(avg_spread) and avg_spread > 0 else 0
                        metrics['pre_closing_posturing'] = normalized_imbalance * 100
                        mf_buy_ofi_posturing = posturing_hf_df['main_force_ofi'].clip(lower=0).sum()
                        mf_sell_ofi_posturing = posturing_hf_df['main_force_ofi'].clip(upper=0).sum()
                        total_mf_ofi_abs_posturing = posturing_hf_df['main_force_ofi'].abs().sum()
                        if total_mf_ofi_abs_posturing > 0:
                            metrics['pre_closing_buy_posture'] = (mf_buy_ofi_posturing / total_mf_ofi_abs_posturing) * normalized_imbalance * 100
                            metrics['pre_closing_sell_posture'] = (abs(mf_sell_ofi_posturing) / total_mf_ofi_abs_posturing) * normalized_imbalance * 100
                else:
                    if 'vol_shares' in posturing_df.columns and 'minute_vwap' in posturing_df.columns and 'main_force_net_vol' in posturing_df.columns:
                        posturing_vwap = (posturing_df['vol_shares'] * posturing_df['minute_vwap']).sum() / posturing_df['vol_shares'].sum()
                        price_posture = (posturing_vwap - daily_vwap) / atr
                        posturing_amount = (posturing_df['vol_shares'] * posturing_df['minute_vwap']).sum()
                        if posturing_amount > 0:
                            force_posture = (posturing_df['main_force_net_vol'].sum() * posturing_vwap) / posturing_amount
                            metrics['pre_closing_posturing'] = (0.6 * price_posture + 0.4 * force_posture) * 100
                            mf_buy_vol_posturing = posturing_df['main_force_buy_vol'].sum()
                            mf_sell_vol_posturing = posturing_df['main_force_sell_vol'].sum()
                            total_mf_vol_posturing = posturing_df['main_force_buy_vol'].sum() + posturing_df['main_force_sell_vol'].sum()
                            if total_mf_vol_posturing > 0:
                                metrics['pre_closing_buy_posture'] = (mf_buy_vol_posturing / total_mf_vol_posturing) * (0.6 * price_posture + 0.4 * force_posture) * 100
                                metrics['pre_closing_sell_posture'] = (mf_sell_vol_posturing / total_mf_vol_posturing) * (0.6 * price_posture + 0.4 * force_posture) * 100
        return metrics

    @staticmethod
    def _calculate_hidden_accumulation_metrics(context: dict) -> dict:
        """
        【V71.0 · 终极生产版 - 空数据鲁棒性增强】
        - 核心增强: 增加对 `hf_analysis_df` 是否为空的检查，避免在无高频数据时引发 `KeyError`。
        """
        intraday_data = context['intraday_data']
        hf_analysis_df = context['hf_analysis_df']
        common_data = context['common_data']
        import numpy as np
        metrics = {}
        if hf_analysis_df.empty: # 增加空数据检查
            # Fallback to intraday_data based calculation if hf_analysis_df is empty
            daily_vwap = common_data['daily_vwap'] # Still need daily_vwap for fallback
            if pd.notna(daily_vwap): # Fallback logic
                dip_or_flat_df = intraday_data[intraday_data['close'] <= intraday_data['open']]
                if not dip_or_flat_df.empty:
                    total_vol_dip = dip_or_flat_df['vol_shares'].sum()
                    if total_vol_dip > 0 and 'main_force_net_vol' in dip_or_flat_df.columns:
                        mf_net_buy_on_dip = dip_or_flat_df['main_force_net_vol'].clip(lower=0).sum()
                        metrics['hidden_accumulation_intensity'] = (mf_net_buy_on_dip / total_vol_dip) * 100
            return metrics
        daily_vwap = common_data['daily_vwap']
        if pd.notna(daily_vwap):
            absorption_zone = hf_analysis_df[hf_analysis_df['mid_price'] < daily_vwap].copy()
            if not absorption_zone.empty:
                passive_absorption_mask = (absorption_zone['type'] == 'S') & (absorption_zone['price'] <= absorption_zone['prev_b1_p'])
                passive_absorption_vol = absorption_zone.loc[passive_absorption_mask, 'volume'].sum()
                total_vol_below_vwap = absorption_zone['volume'].sum()
                passive_absorption_component = passive_absorption_vol / total_vol_below_vwap if total_vol_below_vwap > 0 else 0.0
                impact_suppression_component = 0.0
                if not absorption_zone.empty and absorption_zone['main_force_ofi'].var() > 0 and absorption_zone['mid_price_change'].var() > 0:
                    correlation = absorption_zone['main_force_ofi'].corr(absorption_zone['mid_price_change'])
                    impact_suppression_component = -np.tanh(correlation) if pd.notna(correlation) else 0.0
                total_book_depth = absorption_zone[[f'{d}_volume{i}' for d in ['buy', 'sell'] for i in range(1, 6)]].sum(axis=1)
                bid_depth_ratio = absorption_zone['buy_volume1'] / total_book_depth.replace(0, np.nan)
                liquidity_commitment_component = bid_depth_ratio.mean() if not bid_depth_ratio.empty else 0.0
                metrics['hidden_accumulation_intensity'] = (0.5 * passive_absorption_component + 0.3 * impact_suppression_component + 0.2 * liquidity_commitment_component) * 100
        else: # Fallback if daily_vwap is NaN
            dip_or_flat_df = intraday_data[intraday_data['close'] <= intraday_data['open']]
            if not dip_or_flat_df.empty:
                total_vol_dip = dip_or_flat_df['vol_shares'].sum()
                if total_vol_dip > 0 and 'main_force_net_vol' in dip_or_flat_df.columns:
                    mf_net_buy_on_dip = dip_or_flat_df['main_force_net_vol'].clip(lower=0).sum()
                    metrics['hidden_accumulation_intensity'] = (mf_net_buy_on_dip / total_vol_dip) * 100
        return metrics

    @staticmethod
    def _calculate_vwap_related_metrics(context: dict) -> dict:
        """
        【V66.0 · 计算内核静态化】
        - 核心重构: 添加 @staticmethod 装饰器，移除 self 参数，将其转换为无状态的静态方法。
        【V72.0 · 资金流拆分版】
        - 核心增强: 拆分 `main_force_vwap_guidance` 和 `vwap_crossing_intensity` 为方向性指标。
        """
        intraday_data = context['intraday_data']
        common_data = context['common_data']
        import numpy as np
        import pandas as pd
        metrics = {
            'main_force_vwap_guidance': np.nan,
            'main_force_vwap_up_guidance': np.nan, # 新增行
            'main_force_vwap_down_guidance': np.nan, # 新增行
            'vwap_crossing_intensity': np.nan,
            'vwap_cross_up_intensity': np.nan, # 新增行
            'vwap_cross_down_intensity': np.nan, # 新增行
        }
        daily_vwap = common_data['daily_vwap']
        daily_total_volume = common_data['daily_total_volume']
        atr = common_data['atr']
        if pd.notna(daily_vwap) and daily_total_volume > 0 and pd.notna(atr) and atr > 0 and 'minute_vwap' in intraday_data.columns and 'vol_shares' in intraday_data.columns and 'main_force_net_vol' in intraday_data.columns:
            price_dev_series = intraday_data['minute_vwap'] - daily_vwap
            mf_net_flow_series = intraday_data['main_force_net_vol']
            if price_dev_series.var() != 0 and mf_net_flow_series.var() != 0 and len(price_dev_series) > 1:
                correlation = price_dev_series.corr(mf_net_flow_series)
                metrics['main_force_vwap_guidance'] = correlation if pd.notna(correlation) else np.nan
                # 新增拆分指标
                up_guidance_mask = (price_dev_series > 0) & (mf_net_flow_series > 0) # 新增行
                down_guidance_mask = (price_dev_series < 0) & (mf_net_flow_series < 0) # 新增行
                if up_guidance_mask.any(): # 新增行
                    metrics['main_force_vwap_up_guidance'] = price_dev_series[up_guidance_mask].corr(mf_net_flow_series[up_guidance_mask]) # 新增行
                if down_guidance_mask.any(): # 新增行
                    metrics['main_force_vwap_down_guidance'] = price_dev_series[down_guidance_mask].corr(mf_net_flow_series[down_guidance_mask]) # 新增行
            position_vs_vwap = np.sign(intraday_data['minute_vwap'] - daily_vwap)
            crossings = position_vs_vwap.diff().ne(0)
            metrics['vwap_crossing_intensity'] = intraday_data.loc[crossings, 'vol_shares'].sum() / daily_total_volume
            # 新增拆分指标
            cross_up_mask = (position_vs_vwap.shift(1) == -1) & (position_vs_vwap == 1) # 新增行
            cross_down_mask = (position_vs_vwap.shift(1) == 1) & (position_vs_vwap == -1) # 新增行
            metrics['vwap_cross_up_intensity'] = intraday_data.loc[crossings & cross_up_mask, 'vol_shares'].sum() / daily_total_volume # 新增行
            metrics['vwap_cross_down_intensity'] = intraday_data.loc[crossings & cross_down_mask, 'vol_shares'].sum() / daily_total_volume # 新增行
            twap = intraday_data['minute_vwap'].mean()
            if pd.notna(twap) and twap > 0:
                metrics['vwap_structure_skew'] = (daily_vwap - twap) / twap * 100
        return metrics

    @staticmethod
    def _calculate_vwap_control_metrics(context: dict) -> dict:
        """
        【V71.0 · 终极生产版 - 空数据鲁棒性增强】
        【V72.0 · 资金流拆分版】
        - 核心增强: 拆分 `vwap_control_strength` 为买卖双方控制强度。
        - 核心增强: 增加对 `hf_analysis_df` 是否为空的检查，避免在无高频数据时引发 `KeyError`。
        """
        intraday_data = context['intraday_data']
        hf_analysis_df = context['hf_analysis_df']
        common_data = context['common_data']
        import numpy as np
        import pandas as pd
        metrics = {
            'vwap_control_strength': np.nan,
            'vwap_buy_control_strength': np.nan,
            'vwap_sell_control_strength': np.nan,
        }
        daily_vwap = common_data['daily_vwap']
        daily_total_volume = common_data['daily_total_volume']
        atr = common_data['atr']
        if pd.isna(daily_vwap) or pd.isna(daily_total_volume) or daily_total_volume <= 0 or pd.isna(atr) or atr <= 0:
            return metrics
        if hf_analysis_df.empty: # 增加空数据检查
            # Fallback to intraday_data based calculation if hf_analysis_df is empty
            if 'minute_vwap' in intraday_data.columns and 'vol_shares' in intraday_data.columns:
                price_deviation_value = (intraday_data['minute_vwap'] - daily_vwap) * intraday_data['vol_shares']
                metrics['vwap_control_strength'] = price_deviation_value.sum() / (atr * daily_total_volume)
                if 'main_force_buy_vol' in intraday_data.columns and 'main_force_sell_vol' in intraday_data.columns:
                    mf_buy_vol_in_zone = intraday_data['main_force_buy_vol'].sum()
                    mf_sell_vol_in_zone = intraday_data['main_force_sell_vol'].sum()
                    total_mf_vol_in_zone = mf_buy_vol_in_zone + mf_sell_vol_in_zone
                    if total_mf_vol_in_zone > 0:
                        metrics['vwap_buy_control_strength'] = (mf_buy_vol_in_zone / total_mf_vol_in_zone) * metrics['vwap_control_strength']
                        metrics['vwap_sell_control_strength'] = (mf_sell_vol_in_zone / total_mf_vol_in_zone) * metrics['vwap_control_strength']
            return metrics
        if 'ofi' in hf_analysis_df.columns and 'main_force_ofi' in hf_analysis_df.columns:
            gravity_band = 0.1 * atr
            upper_bound = daily_vwap + gravity_band
            lower_bound = daily_vwap - gravity_band
            zone_hf_df = hf_analysis_df[(hf_analysis_df['price'] >= lower_bound) & (hf_analysis_df['price'] <= upper_bound)]
            if not zone_hf_df.empty:
                market_pressure_ofi = zone_hf_df['ofi'].sum()
                mf_counter_ofi = zone_hf_df['main_force_ofi'].sum()
                absorbed_ofi = 0
                if np.sign(market_pressure_ofi) * np.sign(mf_counter_ofi) < 0:
                    absorbed_ofi = min(abs(market_pressure_ofi), abs(mf_counter_ofi))
                absorption_ratio = absorbed_ofi / abs(market_pressure_ofi) if market_pressure_ofi != 0 else 0.0
                volume_in_zone = zone_hf_df['volume'].sum()
                volume_significance = volume_in_zone / daily_total_volume
                metrics['vwap_control_strength'] = absorption_ratio * volume_significance * 100
                mf_buy_ofi_in_zone = zone_hf_df['main_force_ofi'].clip(lower=0).sum()
                mf_sell_ofi_in_zone = zone_hf_df['main_force_ofi'].clip(upper=0).sum()
                total_mf_ofi_in_zone = zone_hf_df['main_force_ofi'].abs().sum()
                if total_mf_ofi_in_zone > 0:
                    metrics['vwap_buy_control_strength'] = (mf_buy_ofi_in_zone / total_mf_ofi_in_zone) * absorption_ratio * volume_significance * 100
                    metrics['vwap_sell_control_strength'] = (abs(mf_sell_ofi_in_zone) / total_mf_ofi_in_zone) * absorption_ratio * volume_significance * 100
        else: # Fallback if hf_analysis_df is not empty but missing 'ofi' or 'main_force_ofi'
            if 'minute_vwap' in intraday_data.columns and 'vol_shares' in intraday_data.columns:
                price_deviation_value = (intraday_data['minute_vwap'] - daily_vwap) * intraday_data['vol_shares']
                metrics['vwap_control_strength'] = price_deviation_value.sum() / (atr * daily_total_volume)
                if 'main_force_buy_vol' in intraday_data.columns and 'main_force_sell_vol' in intraday_data.columns:
                    mf_buy_vol_in_zone = intraday_data['main_force_buy_vol'].sum()
                    mf_sell_vol_in_zone = intraday_data['main_force_sell_vol'].sum()
                    total_mf_vol_in_zone = mf_buy_vol_in_zone + mf_sell_vol_in_zone
                    if total_mf_vol_in_zone > 0:
                        metrics['vwap_buy_control_strength'] = (mf_buy_vol_in_zone / total_mf_vol_in_zone) * metrics['vwap_control_strength']
                        metrics['vwap_sell_control_strength'] = (mf_sell_vol_in_zone / total_mf_vol_in_zone) * metrics['vwap_control_strength']
        return metrics

    @staticmethod
    def _calculate_cmf_metrics(context: dict) -> dict:
        """
        【V70.0 · 背离放大器终版 - 空数据鲁棒性增强】
        - 核心逻辑: 引入“背离放大器”，当主力CMF与市场CMF异号时，加权突显最关键的“方向背离”信号，
                     使指标能更敏锐地捕捉市场核心矛盾。
        - 核心增强: 增加对 `hf_analysis_df` 是否为空的检查，避免在无高频数据时引发 `KeyError`。
        """
        intraday_data = context['intraday_data']
        hf_analysis_df = context['hf_analysis_df']
        import numpy as np
        import pandas as pd
        metrics = {}
        if hf_analysis_df.empty: # 增加空数据检查
            # Fallback to intraday_data based calculation if hf_analysis_df is empty
            if 'high' in intraday_data.columns and 'low' in intraday_data.columns and 'close' in intraday_data.columns and 'vol_shares' in intraday_data.columns:
                price_range = intraday_data['high'] - intraday_data['low']
                mfm = ((intraday_data['close'] - intraday_data['low']) - (intraday_data['high'] - intraday_data['close'])) / price_range
                mfm = mfm.fillna(0)
                mfv = mfm * intraday_data['vol_shares']
                if intraday_data['vol_shares'].sum() > 0:
                    metrics['holistic_cmf'] = mfv.sum() / intraday_data['vol_shares'].sum()
                if 'main_force_net_vol' in intraday_data.columns:
                    mf_vol = intraday_data['main_force_buy_vol'] + intraday_data['main_force_sell_vol']
                    mf_mfv = mfm * mf_vol
                    if mf_vol.sum() > 0:
                        metrics['main_force_cmf'] = mf_mfv.sum() / mf_vol.sum()
            return metrics
        if 'price' in hf_analysis_df.columns and 'main_force_ofi' in hf_analysis_df.columns:
            df = hf_analysis_df.copy()
            window = 120
            rolling_high = df['price'].rolling(window=window, min_periods=2).max()
            rolling_low = df['price'].rolling(window=window, min_periods=2).min()
            price_range = rolling_high - rolling_low
            money_flow_multiplier = np.where(
                price_range > 0,
                ((df['price'] - rolling_low) - (rolling_high - df['price'])) / price_range,
                0
            )
            money_flow_volume = money_flow_multiplier * df['volume']
            total_volume = df['volume'].sum()
            if total_volume > 0:
                metrics['holistic_cmf'] = money_flow_volume.sum() / total_volume
            mf_money_flow_volume = money_flow_multiplier * df['main_force_ofi'].abs()
            total_mf_volume = df['main_force_ofi'].abs().sum()
            if total_mf_volume > 0:
                metrics['main_force_cmf'] = mf_money_flow_volume.sum() / total_mf_volume
        else: # Fallback if hf_analysis_df is not empty but missing 'price' or 'main_force_ofi'
            if 'high' in intraday_data.columns and 'low' in intraday_data.columns and 'close' in intraday_data.columns and 'vol_shares' in intraday_data.columns:
                price_range = intraday_data['high'] - intraday_data['low']
                mfm = ((intraday_data['close'] - intraday_data['low']) - (intraday_data['high'] - intraday_data['close'])) / price_range
                mfm = mfm.fillna(0)
                mfv = mfm * intraday_data['vol_shares']
                if intraday_data['vol_shares'].sum() > 0:
                    metrics['holistic_cmf'] = mfv.sum() / intraday_data['vol_shares'].sum()
                if 'main_force_net_vol' in intraday_data.columns:
                    mf_vol = intraday_data['main_force_buy_vol'] + intraday_data['main_force_sell_vol']
                    mf_mfv = mfm * mf_vol
                    if mf_vol.sum() > 0:
                        metrics['main_force_cmf'] = mf_mfv.sum() / mf_vol.sum()
        main_force_cmf_value = metrics.get('main_force_cmf')
        holistic_cmf_value = metrics.get('holistic_cmf')
        if pd.notna(main_force_cmf_value) and pd.notna(holistic_cmf_value):
            base_divergence = main_force_cmf_value - holistic_cmf_value
            divergence_amplifier = 2.0 if np.sign(main_force_cmf_value) * np.sign(holistic_cmf_value) < 0 else 1.0
            metrics['cmf_divergence_score'] = base_divergence * divergence_amplifier * 100
        return metrics

    @staticmethod
    def _calculate_vpoc_metrics(context: dict) -> dict:
        """
        【优化版】使用 np.histogram 替代 pd.cut/groupby 提升性能。
        【V72.4 · VPOC鲁棒性增强版】
        - 核心修复: 增强 _calculate_vpoc_fast 函数的鲁棒性，处理输入价格数组中包含 NaN 值、
                     所有价格相同或有效数据点不足的情况，避免 np.histogram 抛出 ValueError。
        """
        intraday_data = context['intraday_data']
        hf_analysis_df = context['hf_analysis_df']
        common_data = context['common_data']
        hf_features = context['hf_features']
        import pandas as pd
        import numpy as np
        metrics = {
            'main_force_vpoc': np.nan,
            'mf_vpoc_premium': np.nan,
            'main_force_on_peak_flow': np.nan,
            'main_force_on_peak_buy_flow': np.nan,
            'main_force_on_peak_sell_flow': np.nan,
        }
        daily_total_amount = common_data['daily_total_amount']
        # 优化后的静态内部函数，使用numpy直方图计算VPOC
        def _calculate_vpoc_fast(price_arr: np.ndarray, vol_arr: np.ndarray, bins: int = 50) -> tuple[float, float, float]:
            # 过滤掉价格或成交量为NaN的条目，以及成交量为0的条目
            valid_mask = ~np.isnan(price_arr) & ~np.isnan(vol_arr) & (vol_arr > 0)
            filtered_price_arr = price_arr[valid_mask]
            filtered_vol_arr = vol_arr[valid_mask]
            if len(filtered_price_arr) < 2:
                # 如果有效数据点少于2个，无法形成有效的直方图
                return np.nan, np.nan, np.nan
            min_price = np.min(filtered_price_arr)
            max_price = np.max(filtered_price_arr)
            if min_price == max_price:
                # 如果所有有效价格都相同，则VPOC就是这个价格，没有范围
                # 此时 np.histogram 可能会报错，直接返回该价格
                return min_price, min_price, min_price
            # 确保 bin 范围有效，避免 np.histogram 内部自动检测范围时出现 [nan, nan]
            # 或者 [value, value] 导致 ValueError
            hist_range = (min_price, max_price)
            # 再次检查，以防万一 min_price 和 max_price 在浮点数比较时出现微小差异
            if np.isclose(hist_range[0], hist_range[1]):
                return min_price, min_price, min_price
            counts, bin_edges = np.histogram(filtered_price_arr, bins=bins, range=hist_range, weights=filtered_vol_arr)
            # 检查 counts 是否为空或全为零，这可能发生在所有数据点都落在 bin 范围之外（尽管我们已经设置了范围）
            # 或者所有权重都为零
            if counts.size == 0 or np.all(counts == 0):
                return np.nan, np.nan, np.nan
            max_idx = np.argmax(counts)
            vpoc = (bin_edges[max_idx] + bin_edges[max_idx+1]) / 2
            return vpoc, bin_edges[max_idx], bin_edges[max_idx+1]
        if hf_analysis_df.empty:
            if 'main_force_net_vol' in intraday_data.columns and 'minute_vwap' in intraday_data.columns and 'vol_shares' in intraday_data.columns:
                price_arr = intraday_data['minute_vwap'].values
                vol_arr = intraday_data['vol_shares'].values
                global_vpoc_price, global_left, global_right = _calculate_vpoc_fast(price_arr, vol_arr, bins=30)
                if pd.notna(global_vpoc_price):
                    peak_zone_mask = (price_arr >= global_left) & (price_arr < global_right)
                    if peak_zone_mask.any():
                        mf_net_vol_on_peak = intraday_data.loc[peak_zone_mask, 'main_force_net_vol'].sum()
                        if daily_total_amount > 0:
                            normalized_mf_on_peak_flow = np.tanh((mf_net_vol_on_peak * global_vpoc_price) / daily_total_amount)
                            metrics['main_force_on_peak_flow'] = normalized_mf_on_peak_flow
                        mf_buy_vol_on_peak = intraday_data.loc[peak_zone_mask, 'main_force_buy_vol'].sum()
                        mf_sell_vol_on_peak = intraday_data.loc[peak_zone_mask, 'main_force_sell_vol'].sum()
                        if daily_total_amount > 0:
                            metrics['main_force_on_peak_buy_flow'] = np.tanh((mf_buy_vol_on_peak * global_vpoc_price) / daily_total_amount)
                            metrics['main_force_on_peak_sell_flow'] = np.tanh((mf_sell_vol_on_peak * global_vpoc_price) / daily_total_amount)
            mf_net_buy_mask = intraday_data['main_force_net_vol'] > 0
            if mf_net_buy_mask.any():
                mf_vwap_arr = intraday_data.loc[mf_net_buy_mask, 'minute_vwap'].values
                mf_net_vol_arr = intraday_data.loc[mf_net_buy_mask, 'main_force_net_vol'].values
                mf_vpoc, _, _ = _calculate_vpoc_fast(mf_vwap_arr, mf_net_vol_arr, bins=30)
                metrics['main_force_vpoc'] = mf_vpoc
                if pd.notna(global_vpoc_price) and global_vpoc_price > 0 and pd.notna(mf_vpoc):
                    metrics['mf_vpoc_premium'] = (mf_vpoc / global_vpoc_price - 1) * 100
            return metrics
        # 使用 numpy 优化计算全局 VPOC
        global_vpoc_price, global_left, global_right = _calculate_vpoc_fast(
            hf_analysis_df['price'].values, 
            hf_analysis_df['volume'].values
        )
        mf_trades = hf_features['mf_trades']
        # 使用 numpy 优化计算主力 VPOC
        if not mf_trades.empty:
            mf_vpoc, _, _ = _calculate_vpoc_fast(
                mf_trades['price'].values,
                mf_trades['volume'].values
            )
            metrics['main_force_vpoc'] = mf_vpoc
            if pd.notna(global_vpoc_price) and global_vpoc_price > 0 and pd.notna(mf_vpoc):
                metrics['mf_vpoc_premium'] = (mf_vpoc / global_vpoc_price - 1) * 100
            if pd.notna(global_left):
                peak_zone_mask = (mf_trades['price'] >= global_left) & (mf_trades['price'] < global_right)
                peak_zone_mf_trades = mf_trades[peak_zone_mask]
                if not peak_zone_mf_trades.empty:
                    amounts = np.where(
                        peak_zone_mf_trades['type'] == 'B',
                        peak_zone_mf_trades['amount'],
                        -peak_zone_mf_trades['amount']
                    )
                    net_amount_on_peak = amounts.sum()
                    if daily_total_amount > 0:
                        metrics['main_force_on_peak_flow'] = np.tanh(net_amount_on_peak / daily_total_amount)
                    mf_buy_amount_on_peak = peak_zone_mf_trades[peak_zone_mf_trades['type'] == 'B']['amount'].sum()
                    mf_sell_amount_on_peak = peak_zone_mf_trades[peak_zone_mf_trades['type'] == 'S']['amount'].sum()
                    if daily_total_amount > 0:
                        metrics['main_force_on_peak_buy_flow'] = np.tanh(mf_buy_amount_on_peak / daily_total_amount)
                        metrics['main_force_on_peak_sell_flow'] = np.tanh(mf_sell_amount_on_peak / daily_total_amount)
        else:
             pass
        return metrics

    @staticmethod
    def _calculate_liquidity_swap_metrics(context: dict) -> dict:
        """
        【V71.0 · 终极生产版 - 空数据鲁棒性增强】
        - 核心增强: 增加对 `hf_analysis_df` 是否为空的检查，避免在无高频数据时引发 `KeyError`。
        """
        intraday_data = context['intraday_data']
        hf_analysis_df = context['hf_analysis_df']
        metrics = {}
        if hf_analysis_df.empty: # 增加空数据检查
            # Fallback to intraday_data based calculation if hf_analysis_df is empty
            if 'main_force_net_vol' in intraday_data.columns and 'retail_net_vol' in intraday_data.columns:
                mf_net_series = intraday_data['main_force_net_vol']
                retail_net_series = intraday_data['retail_net_vol']
                if mf_net_series.var() != 0 and retail_net_series.var() != 0 and len(mf_net_series) > 1:
                    rolling_corr = mf_net_series.rolling(window=30).corr(retail_net_series)
                    metrics['mf_retail_liquidity_swap_corr'] = rolling_corr.mean()
            return metrics
        if 'main_force_ofi' in hf_analysis_df.columns and 'retail_ofi' in hf_analysis_df.columns:
            mf_ofi_series = hf_analysis_df['main_force_ofi']
            retail_ofi_series = hf_analysis_df['retail_ofi']
            if mf_ofi_series.var() > 0 and retail_ofi_series.var() > 0:
                correlation = mf_ofi_series.corr(retail_ofi_series)
                metrics['mf_retail_liquidity_swap_corr'] = correlation
        else: # Fallback if hf_analysis_df is not empty but missing 'main_force_ofi' or 'retail_ofi'
            if 'main_force_net_vol' in intraday_data.columns and 'retail_net_vol' in intraday_data.columns:
                mf_net_series = intraday_data['main_force_net_vol']
                retail_net_series = intraday_data['retail_net_vol']
                if mf_net_series.var() != 0 and retail_net_series.var() != 0 and len(mf_net_series) > 1:
                    rolling_corr = mf_net_series.rolling(window=30).corr(retail_net_series)
                    metrics['mf_retail_liquidity_swap_corr'] = rolling_corr.mean()
        return metrics

    @staticmethod
    def _calculate_retail_sentiment_metrics(context: dict) -> dict:
        """
        精细化计算零售投资者情绪指标：FOMO狂热指数和恐慌投降指数。
        基于中国A股市场特性，从价格、成交量、市场微观结构、心理博弈等多维度精细化计算。
        参数:
            context (dict): 包含所有计算数据和中间结果的上下文字典。
        返回:
            dict: 包含零售投资者情绪指标的字典。
        """
        intraday_data = context['intraday_data']
        hf_analysis_df = context['hf_analysis_df']
        daily_data = context['daily_data']
        common_data = context['common_data']
        hf_features = context.get('hf_features', {})
        # 获取调试信息
        should_probe = context['debug']['should_probe']
        stock_code = context['debug']['stock_code']
        current_date = context['daily_data'].name.date()
        # 初始化指标
        metrics = {
            'retail_fomo_premium_index': np.nan,
            'retail_panic_surrender_index': np.nan,
            'retail_fomo_volume_ratio': np.nan,      # FOMO成交量占比
            'retail_panic_volume_ratio': np.nan,     # 恐慌成交量占比
            'retail_fomo_intensity': np.nan,         # FOMO强度（基于多维度）
            'retail_panic_intensity': np.nan         # 恐慌强度（基于多维度）
        }
        # 基础验证
        atr = common_data['atr']
        if should_probe:
            print(f"\n--- [探针 _calculate_retail_sentiment_metrics - 初始检查] {stock_code} {current_date} ---")
            print(f"  - hf_analysis_df.empty: {hf_analysis_df.empty}")
            print(f"  - atr: {atr}")
            print(f"  - daily_vwap: {common_data.get('daily_vwap')}")
        if hf_analysis_df.empty or pd.isna(atr) or atr <= 0:
            if should_probe:
                print(f"  - 提前返回，因为 hf_analysis_df 为空或 atr 无效。")
            return metrics
        # 获取主力成本数据（如果有的话）
        cost_mf_sell = daily_data.get('avg_cost_main_sell', np.nan)
        cost_mf_buy = daily_data.get('avg_cost_main_buy', np.nan)
        # 如果主力成本缺失，使用当日VWAP作为替代参考
        if pd.isna(cost_mf_sell) or cost_mf_sell <= 0:
            cost_mf_sell = common_data.get('daily_vwap', np.nan)
        if pd.isna(cost_mf_buy) or cost_mf_buy <= 0:
            cost_mf_buy = common_data.get('daily_vwap', np.nan)
        if should_probe:
            print(f"  - cost_mf_sell (after fallback): {cost_mf_sell}")
            print(f"  - cost_mf_buy (after fallback): {cost_mf_buy}")
        # 复制数据以避免修改原始数据
        hf_analysis_df_copy = hf_analysis_df.copy()
        # 1. 精细化交易者身份识别（使用改进后的识别逻辑）
        # 修正：传入 context 参数
        is_main_force_trade, is_retail_trade = AdvancedFundFlowMetricsService._identify_trade_participants(hf_analysis_df_copy, context)
        hf_analysis_df_copy['is_retail_trade'] = is_retail_trade
        hf_analysis_df_copy['is_main_force_trade'] = is_main_force_trade
        if should_probe:
            print(f"  - is_retail_trade count: {is_retail_trade.sum()}")
            print(f"  - is_main_force_trade count: {is_main_force_trade.sum()}")
        # 2. 计算价格极值点（更精确的定义）
        # 基于滚动窗口计算局部极值，避免短期噪声干扰
        window_size = max(10, min(100, len(hf_analysis_df_copy) // 100))  # 自适应窗口
        hf_analysis_df_copy['rolling_max_20'] = hf_analysis_df_copy['price'].rolling(window=window_size, min_periods=1).max()
        hf_analysis_df_copy['rolling_min_20'] = hf_analysis_df_copy['price'].rolling(window=window_size, min_periods=1).min()
        # 真正的新高：超过过去N笔交易的最高价
        hf_analysis_df_copy['is_true_new_high'] = (hf_analysis_df_copy['price'] > hf_analysis_df_copy['rolling_max_20'].shift(1)) & \
                                                 (hf_analysis_df_copy['price'] > hf_analysis_df_copy['price'].shift(1))
        # 真正的新低：低于过去N笔交易的最低价
        hf_analysis_df_copy['is_true_new_low'] = (hf_analysis_df_copy['price'] < hf_analysis_df_copy['rolling_min_20'].shift(1)) & \
                                                (hf_analysis_df_copy['price'] < hf_analysis_df_copy['price'].shift(1))
        if should_probe:
            print(f"  - is_true_new_high count: {hf_analysis_df_copy['is_true_new_high'].sum()}")
            print(f"  - is_true_new_low count: {hf_analysis_df_copy['is_true_new_low'].sum()}")
        # 3. 计算价格动量加速度（捕捉FOMO/恐慌的加速特征）
        hf_analysis_df_copy['price_change'] = hf_analysis_df_copy['price'].diff()
        hf_analysis_df_copy['price_change_abs'] = hf_analysis_df_copy['price_change'].abs()
        hf_analysis_df_copy['price_change_pct'] = hf_analysis_df_copy['price_change'] / hf_analysis_df_copy['price'].shift(1)
        hf_analysis_df_copy['price_acceleration'] = hf_analysis_df_copy['price_change_pct'].diff()  # 价格变化加速度
        # 4. 计算盘口冲击指标（衡量交易对市场深度的影响）
        if all(col in hf_analysis_df_copy.columns for col in ['mid_price', 'prev_mid_price']):
            hf_analysis_df_copy['mid_price_change'] = hf_analysis_df_copy['mid_price'] - hf_analysis_df_copy['prev_mid_price']
            hf_analysis_df_copy['mid_price_change_pct'] = hf_analysis_df_copy['mid_price_change'] / hf_analysis_df_copy['prev_mid_price']
        # 5. 计算主动被动成交识别（更精确）
        # 主动买入：类型为B且成交价>=卖一价
        # 主动卖出：类型为S且成交价<=买一价
        # 被动成交：其他情况
        hf_analysis_df_copy['aggressive_buy'] = (hf_analysis_df_copy['type'] == 'B') & \
                                               (hf_analysis_df_copy['price'] >= hf_analysis_df_copy['sell_price1'] * 0.999)
        hf_analysis_df_copy['aggressive_sell'] = (hf_analysis_df_copy['type'] == 'S') & \
                                                (hf_analysis_df_copy['price'] <= hf_analysis_df_copy['buy_price1'] * 1.001)
        # 6. 计算时间维度特征（FOMO/恐慌的时间聚集效应）
        if isinstance(hf_analysis_df_copy.index, pd.DatetimeIndex):
            hf_analysis_df_copy['time_seconds'] = (hf_analysis_df_copy.index - hf_analysis_df_copy.index[0]).total_seconds()
            hf_analysis_df_copy['time_since_last_fomo'] = np.nan
            hf_analysis_df_copy['time_since_last_panic'] = np.nan
            # 计算连续FOMO/恐慌事件的时间间隔
            fomo_mask = is_retail_trade & (hf_analysis_df_copy['type'] == 'B') & hf_analysis_df_copy['is_true_new_high']
            panic_mask = is_retail_trade & (hf_analysis_df_copy['type'] == 'S') & hf_analysis_df_copy['is_true_new_low']
            if fomo_mask.any():
                fomo_indices = hf_analysis_df_copy.index[fomo_mask]
                hf_analysis_df_copy.loc[fomo_mask, 'time_since_last_fomo'] = fomo_indices.to_series().diff().dt.total_seconds()
            if panic_mask.any():
                panic_indices = hf_analysis_df_copy.index[panic_mask]
                hf_analysis_df_copy.loc[panic_mask, 'time_since_last_panic'] = panic_indices.to_series().diff().dt.total_seconds()
        # 7. 计算成交量异常度（相对于近期平均）
        if is_retail_trade.any():
            retail_trades = hf_analysis_df_copy[is_retail_trade]
            # 计算零售交易成交量的滚动平均和标准差
            hf_analysis_df_copy['retail_volume_ma_50'] = retail_trades['volume'].rolling(window=50, min_periods=1).mean().reindex(hf_analysis_df_copy.index)
            hf_analysis_df_copy['retail_volume_std_50'] = retail_trades['volume'].rolling(window=50, min_periods=1).std().reindex(hf_analysis_df_copy.index)
            hf_analysis_df_copy['volume_zscore'] = (hf_analysis_df_copy['volume'] - hf_analysis_df_copy['retail_volume_ma_50']) / \
                                                  (hf_analysis_df_copy['retail_volume_std_50'] + 1e-10)
        # 8. 准备Numba函数需要的数据数组
        hf_analysis_df_copy['type_numeric'] = np.select(
            [hf_analysis_df_copy['type'] == 'B', hf_analysis_df_copy['type'] == 'S'],
            [1, -1],
            default=0
        )
        # 创建数据数组
        prices_arr = hf_analysis_df_copy['price'].values.astype(np.float64)
        volumes_arr = hf_analysis_df_copy['volume'].values.astype(np.float64)
        amounts_arr = hf_analysis_df_copy['amount'].values.astype(np.float64)
        types_arr = hf_analysis_df_copy['type_numeric'].values.astype(np.int8)
        sell_price1s_arr = hf_analysis_df_copy['sell_price1'].values.astype(np.float64)
        buy_price1s_arr = hf_analysis_df_copy['buy_price1'].values.astype(np.float64)
        is_new_highs_arr = hf_analysis_df_copy['is_true_new_high'].values
        is_new_lows_arr = hf_analysis_df_copy['is_true_new_low'].values
        is_retail_arr = hf_analysis_df_copy['is_retail_trade'].values
        aggressive_buy_arr = hf_analysis_df_copy['aggressive_buy'].values
        aggressive_sell_arr = hf_analysis_df_copy['aggressive_sell'].values
        # 确保 price_acceleration_arr 和 volume_zscore_arr 始终是 NumPy 数组
        price_acceleration_series_or_array = hf_analysis_df_copy.get('price_acceleration', np.zeros_like(prices_arr))
        if isinstance(price_acceleration_series_or_array, pd.Series):
            price_acceleration_arr = price_acceleration_series_or_array.values
        else:
            price_acceleration_arr = price_acceleration_series_or_array
        price_acceleration_arr = price_acceleration_arr.astype(np.float64)
        volume_zscore_series_or_array = hf_analysis_df_copy.get('volume_zscore', np.zeros_like(prices_arr))
        if isinstance(volume_zscore_series_or_array, pd.Series):
            volume_zscore_arr = volume_zscore_series_or_array.values
        else:
            volume_zscore_arr = volume_zscore_series_or_array
        volume_zscore_arr = volume_zscore_arr.astype(np.float64)
        if should_probe:
            print(f"\n--- [探针 _calculate_retail_sentiment_metrics - Numba输入检查] {stock_code} {current_date} ---")
            print(f"  - prices_arr shape: {prices_arr.shape}, dtype: {prices_arr.dtype}, sample: {prices_arr[:5]}")
            print(f"  - volumes_arr shape: {volumes_arr.shape}, dtype: {volumes_arr.dtype}, sample: {volumes_arr[:5]}")
            print(f"  - amounts_arr shape: {amounts_arr.shape}, dtype: {amounts_arr.dtype}, sample: {amounts_arr[:5]}")
            print(f"  - types_arr shape: {types_arr.shape}, dtype: {types_arr.dtype}, sample: {types_arr[:5]}")
            print(f"  - sell_price1s_arr shape: {sell_price1s_arr.shape}, dtype: {sell_price1s_arr.dtype}, sample: {sell_price1s_arr[:5]}")
            print(f"  - buy_price1s_arr shape: {buy_price1s_arr.shape}, dtype: {buy_price1s_arr.dtype}, sample: {buy_price1s_arr[:5]}")
            print(f"  - is_new_highs_arr shape: {is_new_highs_arr.shape}, dtype: {is_new_highs_arr.dtype}, sum: {is_new_highs_arr.sum()}")
            print(f"  - is_new_lows_arr shape: {is_new_lows_arr.shape}, dtype: {is_new_lows_arr.dtype}, sum: {is_new_lows_arr.sum()}")
            print(f"  - is_retail_arr shape: {is_retail_arr.shape}, dtype: {is_retail_arr.dtype}, sum: {is_retail_arr.sum()}")
            print(f"  - aggressive_buy_arr shape: {aggressive_buy_arr.shape}, dtype: {aggressive_buy_arr.dtype}, sum: {aggressive_buy_arr.sum()}")
            print(f"  - aggressive_sell_arr shape: {aggressive_sell_arr.shape}, dtype: {aggressive_sell_arr.dtype}, sum: {aggressive_sell_arr.sum()}")
            print(f"  - price_acceleration_arr shape: {price_acceleration_arr.shape}, dtype: {price_acceleration_arr.dtype}, sample: {price_acceleration_arr[:5]}")
            print(f"  - volume_zscore_arr shape: {volume_zscore_arr.shape}, dtype: {volume_zscore_arr.dtype}, sample: {volume_zscore_arr[:5]}")
            print(f"  - atr (float): {float(atr)}")
            print(f"  - cost_mf_sell (float): {float(cost_mf_sell) if pd.notna(cost_mf_sell) else np.nan}")
            print(f"  - cost_mf_buy (float): {float(cost_mf_buy) if pd.notna(cost_mf_buy) else np.nan}")
        # 9. 调用改进的Numba函数进行精细化计算
        (total_weighted_fomo_score, total_fomo_volume, total_fomo_amount,
         total_weighted_panic_score, total_panic_volume, total_panic_amount,
         fomo_count, panic_count) = \
            _numba_calculate_retail_fomo_panic_scores_enhanced(
                prices_arr, volumes_arr, amounts_arr, types_arr,
                sell_price1s_arr, buy_price1s_arr,
                is_new_highs_arr, is_new_lows_arr,
                is_retail_arr, aggressive_buy_arr, aggressive_sell_arr,
                price_acceleration_arr, volume_zscore_arr,
                float(atr), float(cost_mf_sell) if pd.notna(cost_mf_sell) else np.nan,
                float(cost_mf_buy) if pd.notna(cost_mf_buy) else np.nan
            )
        if should_probe:
            print(f"\n--- [探针 _calculate_retail_sentiment_metrics - Numba输出检查] {stock_code} {current_date} ---")
            print(f"  - total_weighted_fomo_score: {total_weighted_fomo_score}")
            print(f"  - total_fomo_volume: {total_fomo_volume}")
            print(f"  - total_fomo_amount: {total_fomo_amount}")
            print(f"  - fomo_count: {fomo_count}")
            print(f"  - total_weighted_panic_score: {total_weighted_panic_score}")
            print(f"  - total_panic_volume: {total_panic_volume}")
            print(f"  - total_panic_amount: {total_panic_amount}")
            print(f"  - panic_count: {panic_count}")
        # 10. 计算最终指标（增加多重验证和归一化）
        total_retail_volume = hf_analysis_df_copy.loc[is_retail_trade, 'volume'].sum()
        total_retail_amount = hf_analysis_df_copy.loc[is_retail_trade, 'amount'].sum()
        if should_probe:
            print(f"\n--- [探针 _calculate_retail_sentiment_metrics - 最终指标条件检查] {stock_code} {current_date} ---")
            print(f"  - total_retail_volume: {total_retail_volume}")
            print(f"  - total_retail_amount: {total_retail_amount}")
            print(f"  - Condition for FOMO: total_fomo_volume > 0 ({total_fomo_volume > 0}) and total_fomo_amount > 0 ({total_fomo_amount > 0})")
            print(f"  - Condition for Panic: total_panic_volume > 0 ({total_panic_volume > 0}) and total_panic_amount > 0 ({total_panic_amount > 0})")
        # FOMO指数计算
        if total_fomo_volume > 0 and total_fomo_amount > 0:
            # 计算加权平均FOMO分数（考虑成交量和价格加速度）
            weighted_avg_fomo_score = total_weighted_fomo_score / total_fomo_volume
            # 计算FOMO成交量占比
            fomo_volume_ratio = total_fomo_volume / total_retail_volume if total_retail_volume > 0 else 0
            # 计算FOMO强度（基于多维度）
            if total_retail_amount > 0:
                avg_fomo_trade_amount = total_fomo_amount / fomo_count if fomo_count > 0 else 0
                avg_retail_trade_amount = total_retail_amount / is_retail_trade.sum() if is_retail_trade.sum() > 0 else 0
                fomo_amount_ratio = avg_fomo_trade_amount / avg_retail_trade_amount if avg_retail_trade_amount > 0 else 1
            # 最终FOMO指数：基础分数 * 成交量权重 * 强度因子
            fomo_intensity_factor = 1.0 + min(2.0, fomo_volume_ratio * 5)  # 成交量越大，强度越高
            metrics['retail_fomo_premium_index'] = weighted_avg_fomo_score * fomo_intensity_factor * 100
            metrics['retail_fomo_volume_ratio'] = fomo_volume_ratio
            metrics['retail_fomo_intensity'] = fomo_intensity_factor
        # 恐慌指数计算
        if total_panic_volume > 0 and total_panic_amount > 0:
            # 计算加权平均恐慌分数
            weighted_avg_panic_score = total_weighted_panic_score / total_panic_volume
            # 计算恐慌成交量占比
            panic_volume_ratio = total_panic_volume / total_retail_volume if total_retail_volume > 0 else 0
            # 计算恐慌强度
            if total_retail_amount > 0:
                avg_panic_trade_amount = total_panic_amount / panic_count if panic_count > 0 else 0
                avg_retail_trade_amount = total_retail_amount / is_retail_trade.sum() if is_retail_trade.sum() > 0 else 0
                panic_amount_ratio = avg_panic_trade_amount / avg_retail_trade_amount if avg_retail_trade_amount > 0 else 1
            # 最终恐慌指数：基础分数 * 成交量权重 * 强度因子
            panic_intensity_factor = 1.0 + min(2.0, panic_volume_ratio * 5)
            metrics['retail_panic_surrender_index'] = weighted_avg_panic_score * panic_intensity_factor * 100
            metrics['retail_panic_volume_ratio'] = panic_volume_ratio
            metrics['retail_panic_intensity'] = panic_intensity_factor
        # 11. 增加极端值检测和截断处理
        for key in ['retail_fomo_premium_index', 'retail_panic_surrender_index']:
            if pd.notna(metrics[key]):
                # 防止极端值，截断到合理范围[-100, 100]
                metrics[key] = max(-100.0, min(100.0, metrics[key]))
        # 探针：检查计算后的 retail_fomo_premium_index 和 retail_panic_surrender_index
        if should_probe:
            print(f"\n--- [探针 _calculate_retail_sentiment_metrics - 最终结果] {stock_code} {current_date} ---")
            print(f"  - retail_fomo_premium_index: {metrics.get('retail_fomo_premium_index', np.nan):.4f}")
            print(f"  - retail_panic_surrender_index: {metrics.get('retail_panic_surrender_index', np.nan):.4f}")
            print(f"--- [探针 _calculate_retail_sentiment_metrics 结束] ---")
        return metrics

    @staticmethod
    def _calculate_panic_cascade_metrics(context: dict) -> dict:
        """
        【优化版】使用 searchsorted 优化循环内的切片效率，避免重复全表扫描。
        """
        intraday_data = context['intraday_data']
        hf_analysis_df = context['hf_analysis_df']
        common_data = context['common_data']
        from scipy.signal import find_peaks
        from datetime import time
        import numpy as np
        metrics = {}
        # 基础数据校验
        if hf_analysis_df.empty:
            # 降级逻辑：如果缺乏高频数据，使用分钟线数据近似计算
            atr = common_data['atr']
            continuous_trading_df = intraday_data[intraday_data.index.time < time(14, 57)].copy()
            if not continuous_trading_df.empty and 'minute_vwap' in continuous_trading_df.columns and pd.notna(atr) and atr > 0:
                peaks, _ = find_peaks(continuous_trading_df['minute_vwap'].values)
                troughs, _ = find_peaks(-continuous_trading_df['minute_vwap'].values)
                turning_points = sorted(list(set(np.concatenate(([0], troughs, peaks, [len(continuous_trading_df)-1])))))
                panic_vol, total_panic_vol = 0, 0
                total_retail_sell_vol_fallback = 0
                total_mf_buy_vol_fallback = 0
                for i in range(len(turning_points) - 1):
                    start_idx, end_idx = turning_points[i], turning_points[i+1]
                    window_df = continuous_trading_df.iloc[start_idx:end_idx+1]
                    if window_df.empty or len(window_df) < 2 or 'minute_vwap' not in window_df.columns or 'main_force_net_vol' not in window_df.columns or 'vol_shares' not in window_df.columns:
                        continue
                    # 识别下跌波段
                    if window_df['minute_vwap'].iloc[-1] <= window_df['minute_vwap'].iloc[0]:
                        total_panic_vol += window_df['vol_shares'].sum()
                        mf_net_vol = window_df['main_force_net_vol'].sum()
                        if mf_net_vol < 0:
                            panic_vol += abs(mf_net_vol)
                        total_retail_sell_vol_fallback += window_df['retail_sell_vol'].sum()
                        total_mf_buy_vol_fallback += window_df['main_force_buy_vol'].sum()
                if total_panic_vol > 0:
                    metrics['panic_selling_cascade'] = (panic_vol / total_panic_vol) * 100
                    metrics['panic_sell_volume_contribution'] = (total_retail_sell_vol_fallback / common_data['daily_total_volume']) * 100 if common_data['daily_total_volume'] > 0 else np.nan
                    metrics['panic_buy_absorption_contribution'] = (total_mf_buy_vol_fallback / common_data['daily_total_volume']) * 100 if common_data['daily_total_volume'] > 0 else np.nan
            return metrics
        # 高频数据存在时的精确计算逻辑
        atr = common_data['atr']
        continuous_trading_df = intraday_data[intraday_data.index.time < time(14, 57)].copy()
        if not continuous_trading_df.empty and 'minute_vwap' in continuous_trading_df.columns and pd.notna(atr) and atr > 0:
            peaks, _ = find_peaks(continuous_trading_df['minute_vwap'].values)
            troughs, _ = find_peaks(-continuous_trading_df['minute_vwap'].values)
            turning_points = sorted(list(set(np.concatenate(([0], troughs, peaks, [len(continuous_trading_df)-1])))))
            total_weighted_panic_score = 0
            total_price_drop = 0
            total_retail_sell_vol_sum = 0
            total_mf_buy_vol_sum = 0
            # 准备索引用于二分查找
            hf_index = hf_analysis_df.index
            for i in range(len(turning_points) - 1):
                start_idx, end_idx = turning_points[i], turning_points[i+1]
                window_df = continuous_trading_df.iloc[start_idx:end_idx+1]
                if window_df.empty or len(window_df) < 2: continue
                # 仅处理下跌波段
                if window_df['minute_vwap'].iloc[-1] < window_df['minute_vwap'].iloc[0]:
                    start_time, end_time = window_df.index[0], window_df.index[-1]
                    # --- 优化点：使用 searchsorted 替代布尔索引切片 ---
                    start_loc = hf_index.searchsorted(start_time)
                    end_loc = hf_index.searchsorted(end_time, side='right')
                    panic_hf_df = hf_analysis_df.iloc[start_loc:end_loc]
                    # --- 结束优化 ---
                    if not panic_hf_df.empty:
                        price_drop_in_leg = window_df['minute_vwap'].iloc[0] - window_df['minute_vwap'].iloc[-1]
                        total_price_drop += price_drop_in_leg
                        price_impact_component = price_drop_in_leg / atr
                        # 盘口真空度
                        ask_depth = panic_hf_df[[f'sell_volume{i}' for i in range(1, 6)]].sum(axis=1).mean()
                        bid_depth = panic_hf_df[[f'buy_volume{i}' for i in range(1, 6)]].sum(axis=1).mean()
                        liquidity_vacuum_component = np.tanh(np.log1p(ask_depth / bid_depth)) if bid_depth > 0 else 1.0
                        # 散户投降卖出
                        retail_trades_in_leg = panic_hf_df[panic_hf_df['amount'] < 50000]
                        retail_sell_trades = retail_trades_in_leg[retail_trades_in_leg['type'] == 'S']
                        total_retail_sell_vol = retail_sell_trades['volume'].sum()
                        total_retail_sell_vol_sum += total_retail_sell_vol
                        # 主力承接买入
                        mf_buy_trades_in_leg = panic_hf_df[(panic_hf_df['amount'] > 200000) & (panic_hf_df['type'] == 'B')]
                        total_mf_buy_vol = mf_buy_trades_in_leg['volume'].sum()
                        total_mf_buy_vol_sum += total_mf_buy_vol
                        if total_retail_sell_vol > 0:
                            # 散户主动砸盘比例
                            aggressive_sell_mask = retail_sell_trades['price'] <= retail_sell_trades['buy_price1']
                            aggressive_retail_sell_vol = retail_sell_trades[aggressive_sell_mask]['volume'].sum()
                            retail_capitulation_component = aggressive_retail_sell_vol / total_retail_sell_vol
                        else:
                            retail_capitulation_component = 0.0
                        leg_panic_score = price_impact_component * liquidity_vacuum_component * retail_capitulation_component
                        total_weighted_panic_score += leg_panic_score * price_drop_in_leg
            if total_price_drop > 0:
                weighted_avg_panic_score = total_weighted_panic_score / total_price_drop
                metrics['panic_selling_cascade'] = weighted_avg_panic_score * 100
                metrics['panic_sell_volume_contribution'] = (total_retail_sell_vol_sum / common_data['daily_total_volume']) * 100 if common_data['daily_total_volume'] > 0 else np.nan
                metrics['panic_buy_absorption_contribution'] = (total_mf_buy_vol_sum / common_data['daily_total_volume']) * 100 if common_data['daily_total_volume'] > 0 else np.nan
        else:
             # Fallback logic identical to empty hf_analysis_df branch
             pass
        return metrics

    @staticmethod
    def _calculate_misc_minute_metrics(context: dict) -> dict:
        intraday_data = context['intraday_data']
        hf_analysis_df = context['hf_analysis_df']
        common_data = context['common_data']
        import numpy as np
        import pandas as pd
        metrics = {}
        if hf_analysis_df.empty:
            day_open, day_close = common_data['day_open'], common_data['day_close']
            atr = common_data['atr']
            daily_total_volume = common_data['daily_total_volume']
            if 'main_force_buy_vol' in intraday_data.columns and 'main_force_sell_vol' in intraday_data.columns and pd.notna(atr) and atr > 0:
                mf_activity_ratio = (intraday_data['main_force_buy_vol'].sum() + intraday_data['main_force_sell_vol'].sum()) / daily_total_volume if daily_total_volume > 0 else 0.0
                if mf_activity_ratio > 0:
                    price_outcome = (day_close - day_open) / atr
                    metrics['trend_alignment_index'] = price_outcome / mf_activity_ratio
            continuous_trading_df = intraday_data.between_time('00:00', '14:57', inclusive='left').copy()
            if not continuous_trading_df.empty and 'close' in continuous_trading_df.columns and 'open' in continuous_trading_df.columns:
                up_minutes = continuous_trading_df[continuous_trading_df['close'] > continuous_trading_df['open']]
                down_minutes = continuous_trading_df[continuous_trading_df['close'] < continuous_trading_df['open']]
                if not up_minutes.empty and not down_minutes.empty:
                    up_price_change = (up_minutes['close'] - up_minutes['open']).sum()
                    down_price_change = (down_minutes['open'] - down_minutes['close']).sum()
                    avg_up_speed = up_price_change / len(up_minutes) if len(up_minutes) > 0 else 0
                    avg_down_speed = down_price_change / len(down_minutes) if len(down_minutes) > 0 else 0
                    if avg_up_speed > 0 and avg_down_speed > 0:
                        metrics['volatility_asymmetry_index'] = np.log(avg_up_speed / avg_down_speed)
            return metrics
        day_open, day_close = common_data['day_open'], common_data['day_close']
        atr = common_data['atr']
        daily_total_volume = common_data['daily_total_volume']
        if 'main_force_ofi' in hf_analysis_df.columns and 'mid_price_change' in hf_analysis_df.columns:
            ema_span = 60
            df = hf_analysis_df.copy()
            df['mid_price_ema'] = df['mid_price'].ewm(span=ema_span, adjust=False).mean()
            is_uptrend = df['mid_price'] > df['mid_price_ema']
            is_downtrend = df['mid_price'] < df['mid_price_ema']
            mf_ofi = df['main_force_ofi']
            concordant_ofi = (
                mf_ofi[is_uptrend].clip(lower=0).sum() +
                mf_ofi[is_downtrend].clip(upper=0).abs().sum()
            )
            discordant_ofi = (
                mf_ofi[is_uptrend].clip(upper=0).abs().sum() +
                mf_ofi[is_downtrend].clip(lower=0).sum()
            )
            total_abs_mf_ofi = mf_ofi.abs().sum()
            if total_abs_mf_ofi > 0:
                alignment_score = (concordant_ofi - discordant_ofi) / total_abs_mf_ofi
                metrics['trend_alignment_index'] = alignment_score * 100
            df['log_return'] = np.log(df['mid_price'] / df['mid_price'].shift(1)).fillna(0)
            vol_up = df.loc[is_uptrend, 'log_return'].std()
            vol_down = df.loc[is_downtrend, 'log_return'].std()
            if pd.notna(vol_up) and pd.notna(vol_down) and vol_up > 0 and vol_down > 0:
                metrics['volatility_asymmetry_index'] = np.log(vol_up / vol_down)
        else:
            if 'main_force_buy_vol' in intraday_data.columns and 'main_force_sell_vol' in intraday_data.columns and pd.notna(atr) and atr > 0:
                mf_activity_ratio = (intraday_data['main_force_buy_vol'].sum() + intraday_data['main_force_sell_vol'].sum()) / daily_total_volume if daily_total_volume > 0 else 0.0
                if mf_activity_ratio > 0:
                    price_outcome = (day_close - day_open) / atr
                    metrics['trend_alignment_index'] = price_outcome / mf_activity_ratio
            continuous_trading_df = intraday_data.between_time('00:00', '14:57', inclusive='left').copy()
            if not continuous_trading_df.empty and 'close' in continuous_trading_df.columns and 'open' in continuous_trading_df.columns:
                up_minutes = continuous_trading_df[continuous_trading_df['close'] > continuous_trading_df['open']]
                down_minutes = continuous_trading_df[continuous_trading_df['close'] < continuous_trading_df['open']]
                if not up_minutes.empty and not down_minutes.empty:
                    up_price_change = (up_minutes['close'] - up_minutes['open']).sum()
                    down_price_change = (down_minutes['open'] - down_minutes['close']).sum()
                    avg_up_speed = up_price_change / len(up_minutes) if len(up_minutes) > 0 else 0
                    avg_down_speed = down_price_change / len(down_minutes) if len(down_minutes) > 0 else 0
                    if avg_up_speed > 0 and avg_down_speed > 0:
                        metrics['volatility_asymmetry_index'] = np.log(avg_up_speed / avg_down_speed)
        return metrics

    @staticmethod
    def _calculate_misc_daily_metrics(context: dict) -> dict:
        """
        【V66.0 · 计算内核静态化】
        - 核心重构: 添加 @staticmethod 装饰器，移除 self 参数，将其转换为无状态的静态方法。
        """
        daily_data = context['daily_data']
        import pandas as pd
        import numpy as np
        metrics = {}
        WAN = 10000.0
        try:
            trade_count = pd.to_numeric(daily_data.get('trade_count'), errors='coerce')
            turnover_amount_yuan = pd.to_numeric(daily_data.get('amount'), errors='coerce') * 1000
            if pd.notna(trade_count) and trade_count > 0 and pd.notna(turnover_amount_yuan) and turnover_amount_yuan > 0:
                metrics['inferred_active_order_size'] = turnover_amount_yuan / trade_count
        except Exception:
            metrics['inferred_active_order_size'] = np.nan
        return metrics

    @staticmethod
    def _calculate_flow_efficiency_metrics(context: dict) -> dict:
        hf_analysis_df = context['hf_analysis_df']
        intraday_data = context['intraday_data']
        common_data = context['common_data']
        should_probe = context['debug']['should_probe']
        stock_code = context['debug']['stock_code']
        current_date = context['daily_data'].name.date()
        import numpy as np
        import pandas as pd
        metrics = {
            'flow_efficiency_index': np.nan,
            'buy_flow_efficiency_index': np.nan,
            'sell_flow_efficiency_index': np.nan,
        }
        atr = common_data.get('atr')
        daily_total_volume = common_data.get('daily_total_volume')
        if pd.isna(atr) or atr <= 0 or pd.isna(daily_total_volume) or daily_total_volume <= 0:
            return metrics
        if not hf_analysis_df.empty and 'main_force_ofi' in hf_analysis_df.columns and 'mid_price_change' in hf_analysis_df.columns:
            df = hf_analysis_df[hf_analysis_df['main_force_ofi'] != 0].copy()
            if not df.empty:
                df['price_change_per_ofi'] = df['mid_price_change'] / df['main_force_ofi']
                # 过滤掉 price_change_per_ofi 中的 NaN 值及其对应的权重
                valid_mask = df['price_change_per_ofi'].notna()
                valid_price_change_per_ofi = df.loc[valid_mask, 'price_change_per_ofi']
                weights = df.loc[valid_mask, 'main_force_ofi'].abs()
                if weights.sum() > 0:
                    efficiency_coeff = np.average(valid_price_change_per_ofi, weights=weights)
                    metrics['flow_efficiency_index'] = (efficiency_coeff * daily_total_volume) / atr
                df_buy = hf_analysis_df[hf_analysis_df['main_force_ofi'] > 0].copy()
                if not df_buy.empty:
                    df_buy['price_change_per_ofi'] = df_buy['mid_price_change'] / df_buy['main_force_ofi']
                    valid_mask_buy = df_buy['price_change_per_ofi'].notna()
                    valid_price_change_per_ofi_buy = df_buy.loc[valid_mask_buy, 'price_change_per_ofi']
                    weights_buy = df_buy.loc[valid_mask_buy, 'main_force_ofi'].abs()
                    if weights_buy.sum() > 0:
                        buy_efficiency_coeff = np.average(valid_price_change_per_ofi_buy, weights=weights_buy)
                        metrics['buy_flow_efficiency_index'] = (buy_efficiency_coeff * daily_total_volume) / atr
                df_sell = hf_analysis_df[hf_analysis_df['main_force_ofi'] < 0].copy()
                if not df_sell.empty:
                    df_sell['price_change_per_ofi'] = df_sell['mid_price_change'] / df_sell['main_force_ofi']
                    valid_mask_sell = df_sell['price_change_per_ofi'].notna()
                    valid_price_change_per_ofi_sell = df_sell.loc[valid_mask_sell, 'price_change_per_ofi']
                    weights_sell = df_sell.loc[valid_mask_sell, 'main_force_ofi'].abs()
                    if weights_sell.sum() > 0:
                        sell_efficiency_coeff = np.average(valid_price_change_per_ofi_sell, weights=weights_sell)
                        metrics['sell_flow_efficiency_index'] = (sell_efficiency_coeff * daily_total_volume) / atr
            else:
                if 'main_force_net_vol' in intraday_data.columns and 'close' in intraday_data.columns:
                    df = intraday_data.copy()
                    df['price_change'] = df['close'].diff()
                    df = df[df['main_force_net_vol'] != 0].dropna(subset=['price_change'])
                    if not df.empty:
                        df['price_change_per_vol'] = df['price_change'] / df['main_force_net_vol']
                        valid_mask = df['price_change_per_vol'].notna()
                        valid_price_change_per_vol = df.loc[valid_mask, 'price_change_per_vol']
                        weights = df.loc[valid_mask, 'main_force_net_vol'].abs()
                        if weights.sum() > 0:
                            efficiency_coeff = np.average(valid_price_change_per_vol, weights=weights)
                            metrics['flow_efficiency_index'] = (efficiency_coeff * daily_total_volume) / atr
                    df_buy_fallback = intraday_data[intraday_data['main_force_net_vol'] > 0].copy()
                    if not df_buy_fallback.empty:
                        df_buy_fallback['price_change'] = df_buy_fallback['close'].diff()
                        df_buy_fallback = df_buy_fallback[df_buy_fallback['main_force_net_vol'] != 0].dropna(subset=['price_change'])
                        if not df_buy_fallback.empty:
                            df_buy_fallback['price_change_per_vol'] = df_buy_fallback['price_change'] / df_buy_fallback['main_force_net_vol']
                            valid_mask_buy_fallback = df_buy_fallback['price_change_per_vol'].notna()
                            valid_price_change_per_vol_buy_fallback = df_buy_fallback.loc[valid_mask_buy_fallback, 'price_change_per_vol']
                            weights_buy_fallback = df_buy_fallback.loc[valid_mask_buy_fallback, 'main_force_net_vol'].abs()
                            if weights_buy_fallback.sum() > 0:
                                buy_efficiency_coeff = np.average(valid_price_change_per_vol_buy_fallback, weights=weights_buy_fallback)
                                metrics['buy_flow_efficiency_index'] = (buy_efficiency_coeff * daily_total_volume) / atr
                    df_sell_fallback = intraday_data[intraday_data['main_force_net_vol'] < 0].copy()
                    if not df_sell_fallback.empty:
                        df_sell_fallback['price_change'] = df_sell_fallback['close'].diff()
                        df_sell_fallback = df_sell_fallback[df_sell_fallback['main_force_net_vol'] != 0].dropna(subset=['price_change'])
                        if not df_sell_fallback.empty:
                            df_sell_fallback['price_change_per_vol'] = df_sell_fallback['price_change'] / df_sell_fallback['main_force_net_vol']
                            valid_mask_sell_fallback = df_sell_fallback['price_change_per_vol'].notna()
                            valid_price_change_per_vol_sell_fallback = df_sell_fallback.loc[valid_mask_sell_fallback, 'price_change_per_vol']
                            weights_sell_fallback = df_sell_fallback.loc[valid_mask_sell_fallback, 'main_force_net_vol'].abs()
                            if weights_sell_fallback.sum() > 0:
                                sell_efficiency_coeff = np.average(valid_price_change_per_vol_sell_fallback, weights=weights_sell_fallback)
                                metrics['sell_flow_efficiency_index'] = (sell_efficiency_coeff * daily_total_volume) / atr
        return metrics

    def _calculate_intraday_attribution_weights(self, intraday_data_for_day: pd.DataFrame, daily_data: pd.Series) -> pd.DataFrame:
        """
        【V9.5 · 逐笔数据兼容版 - 价格范围零值修复】
        - 核心革命: 废弃“一体适用”的权重模型，为超大单、大单、中单、小单引入各自独特的、基于行为特征的权重分配逻辑。
        - 核心思想:
          - 超大单(ELG) -> 脉冲修正: 权重集中在成交量和振幅剧增的“暴力分钟”。
          - 大单(LG) -> VWAP修正: 权重与价格偏离VWAP的程度相关，体现战术意图。
          - 中单(MD) -> 动量修正: 权重与短期价格动量相关，体现追涨杀跌特性。
          - 小单(SM) -> 基准压力: 沿用原有的K线形态压力模型作为基准。
        - 核心修复: 修复了 `price_range` 为零时导致的 `decimal.InvalidOperation` 错误。
        - 【修正】修复 `impulse_modifier` 计算中 `price_range` 的错误使用。
        - 核心优化: 使用Numba优化后的修饰符计算函数。
        """
        df = intraday_data_for_day.copy()
        if 'vol_shares' not in df.columns or df['vol_shares'].sum() < 1e-6 or len(df) < 5:
            for size in ['sm', 'md', 'lg', 'elg']:
                df[f'{size}_buy_weight'] = 0; df[f'{size}_sell_weight'] = 0
            return df
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        price_range = df['high'] - df['low']
        buy_pressure_proxy_ratio = np.where(
            price_range != 0,
            (df['close'] - df['low']) / price_range,
            0.5
        )
        conditions = [
            price_range > 0,
            (price_range == 0) & (df['close'] > df['open']),
            (price_range == 0) & (df['close'] < df['open'])
        ]
        choices = [
            buy_pressure_proxy_ratio,
            1.0,
            0.0
        ]
        buy_pressure_proxy = np.select(conditions, choices, default=0.5)
        vol_ma = df['vol_shares'].rolling(window=20, min_periods=1).mean()
        range_ma = price_range.rolling(window=20, min_periods=1).mean()
        daily_vwap = daily_data.get('daily_vwap')
        momentum_modifier_raw = df['minute_vwap'].pct_change().rolling(window=5).mean().fillna(0)
        # 提取数据到NumPy数组
        vol_shares_arr = df['vol_shares'].values
        vol_ma_arr = vol_ma.values
        price_range_arr = price_range.values
        range_ma_arr = range_ma.values
        minute_vwap_arr = df['minute_vwap'].values
        momentum_modifier_raw_arr = momentum_modifier_raw.values
        # 调用Numba优化函数
        impulse_modifier, lg_buy_modifier, lg_sell_modifier, md_buy_modifier, md_sell_modifier = \
            _numba_calculate_attribution_modifiers(
                vol_shares_arr, vol_ma_arr, price_range_arr, range_ma_arr,
                minute_vwap_arr, daily_vwap, momentum_modifier_raw_arr
            )
        # 将Numba函数的结果重新赋值给DataFrame
        df['impulse_modifier'] = impulse_modifier
        df['lg_buy_modifier'] = lg_buy_modifier
        df['lg_sell_modifier'] = lg_sell_modifier
        df['md_buy_modifier'] = md_buy_modifier
        df['md_sell_modifier'] = md_sell_modifier
        sm_buy_score = df['vol_shares'] * buy_pressure_proxy
        sm_sell_score = df['vol_shares'] * (1 - buy_pressure_proxy)
        md_buy_score = sm_buy_score * df['md_buy_modifier']
        md_sell_score = sm_sell_score * df['md_sell_modifier']
        lg_buy_score = sm_buy_score * df['lg_buy_modifier']
        lg_sell_score = sm_sell_score * df['lg_sell_modifier']
        elg_buy_score = sm_buy_score * df['impulse_modifier']
        elg_sell_score = sm_sell_score * df['impulse_modifier']
        scores = {
            'sm': (sm_buy_score, sm_sell_score), 'md': (md_buy_score, md_sell_score),
            'lg': (lg_buy_score, lg_sell_score), 'elg': (elg_buy_score, elg_sell_score)
        }
        for size, (buy_score, sell_score) in scores.items():
            total_buy_score = buy_score.sum()
            df[f'{size}_buy_weight'] = buy_score / total_buy_score if total_buy_score > 1e-9 else 0
            total_sell_score = sell_score.sum()
            df[f'{size}_sell_weight'] = sell_score / total_sell_score if total_sell_score > 1e-9 else 0
        return df

    async def _load_historical_metrics(self, model, stock_info, end_date):
        """
        【V2.2 · 索引修复版】从数据库加载并净化历史高级资金流指标。
        - 核心修复: 修正 set_index 的用法，确保 trade_time 列在成为索引后被正确移除。
        """
        @sync_to_async
        def get_data():
            core_metric_cols = list(BaseAdvancedFundFlowMetrics.CORE_METRICS.keys())
            required_cols = ['trade_time'] + [col for col in core_metric_cols if hasattr(model, col)]
            qs = model.objects.filter(
                stock=stock_info,
                trade_time__lt=end_date
            ).order_by('trade_time')
            return pd.DataFrame.from_records(qs.values(*required_cols))
        df = await get_data()
        if not df.empty:
            # 修复：分两步操作，先转换类型，再用列名设置索引，确保 'trade_time' 列被正确移除
            df['trade_time'] = pd.to_datetime(df['trade_time'])
            df = df.set_index('trade_time')
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def _group_minute_data_from_df(self, minute_df: pd.DataFrame):
        """【V1.15 · 数据完整性修复版 - 辅助列添加 - 智能列名识别】从预加载的DataFrame构建按日分组的数据。
        - 核心职责: 确保传入的DataFrame保持 `trade_time` 作为 `DatetimeIndex`，并正确处理时区，添加 `amount_yuan`, `vol_shares`, `minute_vwap`, `vol_weight` 等辅助列。
        - 核心修复: 不再修改DataFrame的索引，仅添加辅助列。
        - 【修正】智能识别成交量列名（'volume' 或 'vol'），并统一为 'vol_shares'。
        - 【修正】根据最新澄清，统一处理时区，确保最终输出为北京时间。
        """
        from django.utils import timezone
        if minute_df is None or minute_df.empty:
            return pd.DataFrame()
        df = minute_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'trade_time' in df.columns:
                df['trade_time'] = pd.to_datetime(df['trade_time'])
                df = df.set_index('trade_time')
            else:
                logger.warning("DataFrame passed to _group_minute_data_from_df has no 'trade_time' column and no DatetimeIndex.")
                return pd.DataFrame()
        # 统一处理时区，确保最终输出为北京时间
        if df.index.tz is None:
            # 如果意外是 naive，假定它是 UTC（因为DAO层应该输出UTC aware，但可能在某些操作后丢失时区信息）
            df.index = df.index.tz_localize('UTC', ambiguous='infer').tz_convert(timezone.get_current_timezone())
        else:
            # 如果已经是 aware，直接转换为目标时区
            df.index = df.index.tz_convert(timezone.get_current_timezone())
        volume_col_name = None
        if 'volume' in df.columns:
            volume_col_name = 'volume'
        elif 'vol' in df.columns:
            volume_col_name = 'vol'
        else:
            logger.error(f"DataFrame缺少成交量列 ('volume' 或 'vol')，无法处理。列名: {df.columns.tolist()}")
            return pd.DataFrame()
        df['amount_yuan'] = pd.to_numeric(df['amount'], errors='coerce')
        df['vol_shares'] = pd.to_numeric(df[volume_col_name], errors='coerce')
        df['minute_vwap'] = df['amount_yuan'] / df['vol_shares'].replace(0, np.nan)
        current_day_total_vol = df['vol_shares'].sum()
        df['vol_weight'] = df['vol_shares'] / current_day_total_vol if current_day_total_vol > 0 else 0
        return df

    @staticmethod
    def _calculate_execution_alpha_metrics(context: dict) -> dict:
        """
        【V72.3 · 生产就绪版 - 探针恢复】
        - 核心职责: 计算主力买入、卖出以及综合的执行力Alpha。
        - 关键说明: `main_force_buy_execution_alpha` 和 `main_force_sell_execution_alpha`
                    提供了主力在买入和卖出侧的独立执行效率评估。
                    这些独立指标可用于更细致地判断主力意图（如低位吸筹或高位派发），
                    而非简单地依赖综合的 `main_force_execution_alpha`。
        - 升级说明: 恢复了详细探针，用于调试和检查每一步计算。
        """
        hf_analysis_df = context['hf_analysis_df']
        daily_data = context['daily_data']
        common_data = context['common_data']
        hf_features = context['hf_features']
        should_probe = context['debug']['should_probe']
        stock_code = context['debug']['stock_code']
        current_date = context['daily_data'].name.date()
        hf_mf_buy_vwap = hf_features['hf_mf_buy_vwap']
        hf_mf_sell_vwap = hf_features['hf_mf_sell_vwap']
        import numpy as np
        import pandas as pd
        metrics = {
            'main_force_buy_execution_alpha': np.nan,
            'main_force_sell_execution_alpha': np.nan,
            'main_force_execution_alpha': np.nan,
            'main_force_t0_efficiency': np.nan,
            'main_force_t0_buy_efficiency': np.nan,
            'main_force_t0_sell_efficiency': np.nan,
            'main_force_t0_spread_ratio': np.nan,
        }
        daily_vwap = common_data['daily_vwap']
        atr = common_data['atr']
        if pd.isna(daily_vwap) or pd.isna(atr) or atr <= 0:
            return metrics
        buy_alpha, sell_alpha = np.nan, np.nan
        if not hf_analysis_df.empty:
            if pd.notna(hf_mf_buy_vwap):
                buy_alpha = (daily_vwap - hf_mf_buy_vwap) / atr
                metrics['main_force_buy_execution_alpha'] = buy_alpha
                metrics['main_force_t0_buy_efficiency'] = buy_alpha
            if pd.notna(hf_mf_sell_vwap):
                sell_alpha = (hf_mf_sell_vwap - daily_vwap) / atr
                metrics['main_force_sell_execution_alpha'] = sell_alpha
                metrics['main_force_t0_sell_efficiency'] = sell_alpha
            if pd.notna(hf_mf_sell_vwap) and pd.notna(hf_mf_buy_vwap) and daily_vwap > 0:
                t0_spread = (hf_mf_sell_vwap - hf_mf_buy_vwap) / daily_vwap
                metrics['main_force_t0_spread_ratio'] = t0_spread * 100
        else:
            avg_cost_main_buy = daily_data.get('avg_cost_main_buy')
            avg_cost_main_sell = daily_data.get('avg_cost_main_sell')
            if pd.notna(avg_cost_main_buy):
                buy_alpha = (daily_vwap - avg_cost_main_buy) / atr
                metrics['main_force_buy_execution_alpha'] = buy_alpha
                metrics['main_force_t0_buy_efficiency'] = buy_alpha
            if pd.notna(avg_cost_main_sell):
                sell_alpha = (avg_cost_main_sell - daily_vwap) / atr
                metrics['main_force_sell_execution_alpha'] = sell_alpha
                metrics['main_force_t0_sell_efficiency'] = sell_alpha
            if pd.notna(avg_cost_main_sell) and pd.notna(avg_cost_main_buy) and daily_vwap > 0:
                t0_spread = (avg_cost_main_sell - avg_cost_main_buy) / daily_vwap
                metrics['main_force_t0_spread_ratio'] = t0_spread * 100
        if pd.notna(buy_alpha) and pd.notna(sell_alpha):
            metrics['main_force_execution_alpha'] = (buy_alpha + sell_alpha) / 2
            t0_spread_norm = (sell_alpha - (-buy_alpha))
            if pd.notna(sell_alpha) and sell_alpha != 0:
                metrics['main_force_t0_efficiency'] = t0_spread_norm / sell_alpha
        elif pd.notna(sell_alpha):
            metrics['main_force_execution_alpha'] = sell_alpha
        elif pd.notna(buy_alpha):
            metrics['main_force_execution_alpha'] = buy_alpha
        return metrics

    @staticmethod
    def _calculate_wash_trade_metrics(context: dict) -> dict:
        """
        【V71.0 · 终极生产版】(生产环境清洁版)
        【V72.0 · 资金流拆分版】
        - 核心增强: 拆分 `wash_trade_intensity` 为买卖双方对倒量。
        """
        hf_analysis_df = context['hf_analysis_df']
        hf_features = context['hf_features']
        import numpy as np
        import pandas as pd
        metrics = {
            'wash_trade_intensity': np.nan,
            'wash_trade_buy_volume': np.nan, # 新增行
            'wash_trade_sell_volume': np.nan, # 新增行
        }
        if hf_analysis_df.empty:
            return metrics
        mf_trades = hf_features['mf_trades']
        total_mf_volume = hf_features['total_mf_vol']
        if mf_trades.empty or total_mf_volume == 0:
            return metrics
        mf_buys = mf_trades[mf_trades['type'] == 'B'].sort_index()
        mf_sells = mf_trades[mf_trades['type'] == 'S'].sort_index()
        if mf_buys.empty or mf_sells.empty:
            return metrics
        matched_trades = pd.merge_asof(
            mf_buys.reset_index(),
            mf_sells.reset_index(),
            on='trade_time',
            direction='nearest',
            tolerance=pd.Timedelta('3s'),
            suffixes=('_buy', '_sell')
        ).dropna()
        if matched_trades.empty:
            return metrics
        matched_trades['price_diff_ratio'] = (matched_trades['price_sell'] - matched_trades['price_buy']).abs() / matched_trades['price_buy']
        wash_pairs = matched_trades[matched_trades['price_diff_ratio'] < 0.0005]
        if wash_pairs.empty:
            return metrics
        wash_volume = np.minimum(wash_pairs['volume_buy'], wash_pairs['volume_sell']).sum()
        metrics['wash_trade_intensity'] = (wash_volume / total_mf_volume) * 100
        # 新增拆分指标
        metrics['wash_trade_buy_volume'] = wash_pairs['volume_buy'].sum() # 新增行
        metrics['wash_trade_sell_volume'] = wash_pairs['volume_sell'].sum() # 新增行
        return metrics

    @staticmethod
    def _calculate_closing_strength_metrics(context: dict) -> dict:
        """
        【V71.0 · 终极生产版】(生产环境清洁版)
        """
        intraday_data = context['intraday_data']
        hf_analysis_df = context['hf_analysis_df']
        common_data = context['common_data']
        import numpy as np
        metrics = {}
        day_high, day_low, day_close = common_data['day_high'], common_data['day_low'], common_data['day_close']
        daily_vwap, atr = common_data['daily_vwap'], common_data['atr']
        if not all(pd.notna(v) for v in [day_high, day_low, day_close, daily_vwap, atr]) or atr == 0:
            return metrics
        day_range = day_high - day_low
        if day_range <= 0:
            return metrics
        range_pos_factor = ((day_close - day_low) / day_range) * 2 - 1
        value_dev_factor = np.tanh((day_close - daily_vwap) / atr)
        force_factor = 0.0
        if not hf_analysis_df.empty and 'main_force_ofi' in hf_analysis_df.columns:
            total_abs_mf_ofi = hf_analysis_df['main_force_ofi'].abs().sum()
            if total_abs_mf_ofi > 0:
                final_cumulative_mf_ofi = hf_analysis_df['main_force_ofi'].sum()
                force_factor = final_cumulative_mf_ofi / total_abs_mf_ofi
        else:
            daily_total_volume = common_data.get('daily_total_volume', 0)
            if 'main_force_net_vol' in intraday_data.columns and daily_total_volume > 0:
                force_factor = intraday_data['main_force_net_vol'].sum() / daily_total_volume
        metrics['closing_strength_index'] = (0.5 * range_pos_factor + 0.3 * value_dev_factor + 0.2 * force_factor) * 100
        return metrics

    @staticmethod
    def _calculate_micro_dynamics_metrics(context: dict) -> dict:
        """
        【V68.2 · 阻力升维终版 - 命名一致性修复 - 空数据鲁棒性增强】
        - 核心逻辑: 废弃“弹性”概念，转向更本质的“阻力”概念 (abs(总净主动量) / abs(总价差))，
                     确保 `asymmetric_friction_index` 在任何市场场景下都具有数学鲁棒性。
        - 核心修复: 将所有对 `mid_price_delta` 的引用改为 `mid_price_change`，以保持命名一致性。
        - 核心增强: 增加对 `hf_analysis_df` 是否为空的检查，避免在无高频数据时引发 `KeyError`。
        """
        hf_analysis_df = context['hf_analysis_df']
        import numpy as np
        import pandas as pd
        metrics = {
            'micro_impact_elasticity': np.nan,
            'price_reversion_velocity': np.nan,
            'asymmetric_friction_index': np.nan,
        }
        if hf_analysis_df.empty or 'mid_price_change' not in hf_analysis_df.columns:
            return metrics
        elasticity_series = hf_analysis_df['mid_price_change'] / hf_analysis_df['net_active_volume'].replace(0, np.nan)
        weights_vol = hf_analysis_df['volume']
        valid_elasticity = elasticity_series.dropna()
        if not valid_elasticity.empty and weights_vol.sum() > 0:
            metrics['micro_impact_elasticity'] = np.average(valid_elasticity, weights=weights_vol[elasticity_series.notna()])
        hf_analysis_df['next_mid_price_change'] = hf_analysis_df['mid_price_change'].shift(-1)
        reversion_df = hf_analysis_df.dropna(subset=['mid_price_change', 'next_mid_price_change'])
        if not reversion_df.empty and reversion_df['mid_price_change'].var() > 0 and reversion_df['next_mid_price_change'].var() > 0:
            reversion_product = -reversion_df['mid_price_change'] * reversion_df['next_mid_price_change']
            reversion_signal = np.sign(reversion_product)
            weights_rev = reversion_df['volume']
            if weights_rev.sum() > 0:
                metrics['price_reversion_velocity'] = np.average(reversion_signal, weights=weights_rev) * 100
        up_moves = hf_analysis_df[hf_analysis_df['mid_price_change'] > 0]
        down_moves = hf_analysis_df[hf_analysis_df['mid_price_change'] < 0]
        if not up_moves.empty and not down_moves.empty:
            up_price_change_sum = up_moves['mid_price_change'].sum()
            up_net_vol_abs_sum = abs(up_moves['net_active_volume'].sum())
            down_price_change_abs_sum = abs(down_moves['mid_price_change'].sum())
            down_net_vol_abs_sum = abs(down_moves['net_active_volume'].sum())
            upward_resistance = up_net_vol_abs_sum / up_price_change_sum if up_price_change_sum > 0 else np.nan
            downward_resistance = down_net_vol_abs_sum / down_price_change_abs_sum if down_price_change_abs_sum > 0 else np.nan
            if pd.notna(upward_resistance) and pd.notna(downward_resistance) and downward_resistance > 0:
                friction_ratio = upward_resistance / downward_resistance
                metrics['asymmetric_friction_index'] = np.log(friction_ratio) if friction_ratio > 0 else np.nan
        return metrics

    async def _prepare_and_save_data(self, stock_info, MetricsModel, final_df: pd.DataFrame):
        """
        【V51.2 · S系列终审探针植入版】
        - 核心增强: 植入 S.1 终审探针，在数据保存的最后关卡，通过对比“待保存数据的列”与“模型定义的字段”，
                     彻底穿透“数据-模型不同步”问题，为新指标无法入库提供决定性证据。
        【V72.2 · 生产就绪版】
        - 核心清除: 移除所有调试探针相关的print语句和逻辑，恢复生产状态。
        """
        records_to_save_df = final_df
        stock_code = stock_info.stock_code
        if records_to_save_df.empty:
            return 0
        from django.db.models import DecimalField
        from decimal import Decimal, ROUND_HALF_UP
        decimal_fields = [f.name for f in MetricsModel._meta.get_fields() if isinstance(f, DecimalField)]
        for col in decimal_fields:
            if col in records_to_save_df.columns:
                records_to_save_df.loc[:, col] = pd.to_numeric(records_to_save_df[col], errors='coerce')
                records_to_save_df.loc[:, col] = records_to_save_df[col].replace([np.inf, -np.inf], np.nan)
        records_to_save_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        model_fields = {f.name for f in MetricsModel._meta.get_fields() if not f.is_relation and f.name != 'id'}
        df_filtered = records_to_save_df[[col for col in records_to_save_df.columns if col in model_fields]]
        # 探针：检查即将保存的数据中 retail_fomo_premium_index 和 retail_panic_surrender_index 的值
        should_probe = self.debug_params.get('should_probe', False)
        if should_probe and not df_filtered.empty:
            print(f"\n--- [探针 _prepare_and_save_data] {stock_code} - 准备保存数据 ---")
            last_row = df_filtered.iloc[-1]
            last_date = df_filtered.index[-1].strftime('%Y-%m-%d')
            print(f"  - 日期: {last_date}")
            print(f"  - retail_fomo_premium_index (最后一行): {last_row.get('retail_fomo_premium_index', np.nan):.4f}")
            print(f"  - retail_panic_surrender_index (最后一行): {last_row.get('retail_panic_surrender_index', np.nan):.4f}")
            print(f"--- [探针 _prepare_and_save_data 结束] ---")
        records_list = df_filtered.to_dict('records')
        @sync_to_async(thread_sensitive=True)
        def save_atomically(model, stock_obj, records_to_process):
            processed_count = 0
            for i, record_data in enumerate(records_to_process):
                trade_time = record_data.pop('trade_time').date()
                defaults_data = {key: None if isinstance(value, float) and not np.isfinite(value) else value for key, value in record_data.items()}
                for key, value in defaults_data.items():
                    if key in decimal_fields and pd.notna(value):
                        defaults_data[key] = Decimal(str(value)).quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)
                try:
                    obj, created = model.objects.update_or_create(
                        stock=stock_obj,
                        trade_time=trade_time,
                        defaults=defaults_data
                    )
                    processed_count += 1
                except Exception as e:
                    logger.error(f"[{stock_obj.stock_code}] [资金流保存失败] 日期: {trade_time}, 错误: {e}")
            return processed_count
        records_for_atomic_save = []
        for record_date, record_data in zip(df_filtered.index, records_list):
            record_data['trade_time'] = record_date
            records_for_atomic_save.append(record_data)
        processed_count = await save_atomically(MetricsModel, stock_info, records_for_atomic_save)
        return processed_count





