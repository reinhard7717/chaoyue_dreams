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

class AdvancedFundFlowMetricsService:
    """
    【V1.0 · 兵工厂模式】高级资金流指标服务
    - 核心职责: 封装所有高级资金流指标的加载、计算、融合与存储逻辑。
    - 架构优势: 实现业务逻辑与任务调度的完全解耦。
    """
    def __init__(self, debug_params: dict = None): # 新增 debug_params 参数
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
        all_new_core_metrics_df = pd.DataFrame()
        for i in range(0, len(dates_to_process), CHUNK_SIZE):
            chunk_dates = dates_to_process[i:i + CHUNK_SIZE]
            if chunk_dates.empty:
                continue
            chunk_start_date, chunk_end_date = chunk_dates.min(), chunk_dates.max()
            chunk_raw_data_df = await self._load_and_merge_sources(stock_info, start_date=chunk_start_date, end_date=chunk_end_date)
            if chunk_raw_data_df.empty:
                continue
            # 核心修正：移除独立的 daily_vwap 计算步骤，将其整合到核心合成方法中
            self._minute_df_daily_grouped = await self._get_daily_grouped_minute_data(stock_info, chunk_raw_data_df.index)
            chunk_new_metrics_df, _, _ = self._synthesize_and_forge_metrics(stock_code, chunk_raw_data_df)
            all_new_core_metrics_df = pd.concat([all_new_core_metrics_df, chunk_new_metrics_df])
        if hasattr(self, '_minute_df_daily_grouped'):
            del self._minute_df_daily_grouped
        if all_new_core_metrics_df.empty:
            return 0
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

    def _engineer_hf_features(self, raw_hf_df: pd.DataFrame, daily_total_volume: float) -> tuple[pd.DataFrame, dict]:
        """
        【V64.3 · 主力订单流失衡列名修复版】
        - 核心修复: 解决 `KeyError: 'main_force_ofi'`。
                    现在，`hf_analysis_df['main_force_ofi']` 和 `hf_analysis_df['retail_ofi']`
                    被明确地定义为基于 `net_active_volume` (实际执行的净主动成交量) 的逐笔订单流失衡。
                    这确保了下游方法能够正确访问这些列，并反映主力/散户的实际成交行为。
        - 核心思想: `main_force_ofi` 指标名称保持不变，但其底层数据源已从挂单量失衡（`ofi`）
                    切换为实际执行的净主动成交量（`net_active_volume`），并根据交易金额进行主力/散户划分。
        """
        import numpy as np
        features = {
            'mf_trades': pd.DataFrame(), 'buy_trades_mask': pd.Series(dtype=bool),
            'sell_trades_mask': pd.Series(dtype=bool), 'total_mf_vol': 0.0,
            'mf_buy_vol': 0.0, 'mf_sell_vol': 0.0, 'offensive_volume': 0.0,
            'passive_volume': 0.0, 'hf_mf_buy_vwap': np.nan, 'hf_mf_sell_vwap': np.nan,
        }
        if raw_hf_df is None or raw_hf_df.empty:
            return pd.DataFrame(), features
        hf_analysis_df = raw_hf_df.copy()
        hf_analysis_df['mid_price'] = (hf_analysis_df['buy_price1'] + hf_analysis_df['sell_price1']) / 2
        hf_analysis_df['prev_mid_price'] = hf_analysis_df['mid_price'].shift(1)
        hf_analysis_df['mid_price_delta'] = hf_analysis_df['mid_price'].diff() # 确保 mid_price_delta 存在
        # --- 保留原始的基于挂单量的OFI，用于需要盘口压力的指标 ---
        buy_pressure_quote = np.where(hf_analysis_df['mid_price'] >= hf_analysis_df['prev_mid_price'], hf_analysis_df['buy_volume1'].shift(1), 0)
        sell_pressure_quote = np.where(hf_analysis_df['mid_price'] <= hf_analysis_df['prev_mid_price'], hf_analysis_df['sell_volume1'].shift(1), 0)
        hf_analysis_df['ofi'] = buy_pressure_quote - sell_pressure_quote # 这是基于挂单量的OFI
        is_main_force_trade = hf_analysis_df['amount'] > 200000
        is_retail_trade = hf_analysis_df['amount'] < 50000
        # --- 计算净主动成交量 (实际执行的买卖成交量) ---
        active_buy_mask = hf_analysis_df['price'] >= hf_analysis_df['sell_price1']
        active_sell_mask = hf_analysis_df['price'] <= hf_analysis_df['buy_price1']
        net_active_volume_series = pd.Series(0.0, index=hf_analysis_df.index)
        net_active_volume_series.loc[active_buy_mask] = hf_analysis_df.loc[active_buy_mask, 'volume']
        net_active_volume_series.loc[active_sell_mask] = -hf_analysis_df.loc[active_sell_mask, 'volume']
        hf_analysis_df['net_active_volume'] = net_active_volume_series
        # --- 修复：将基于实际执行的净主动成交量赋值给 'main_force_ofi' 和 'retail_ofi' 列 ---
        hf_analysis_df['main_force_ofi'] = np.where(is_main_force_trade, hf_analysis_df['net_active_volume'], 0)
        hf_analysis_df['retail_ofi'] = np.where(is_retail_trade, hf_analysis_df['net_active_volume'], 0)
        hf_analysis_df['mid_price_change'] = hf_analysis_df['mid_price'].diff()
        if 'volume_realtime' in hf_analysis_df.columns and 'snapshot_time' in hf_analysis_df.columns:
            snapshot_changed_mask = hf_analysis_df['snapshot_time'] != hf_analysis_df['snapshot_time'].shift(1)
            volume_delta = hf_analysis_df['volume_realtime'].diff().fillna(0)
            hf_analysis_df['market_vol_delta'] = np.where(snapshot_changed_mask, volume_delta, 0)
        hf_analysis_df['prev_a1_p'] = hf_analysis_df['sell_price1'].shift(1)
        hf_analysis_df['prev_b1_p'] = hf_analysis_df['buy_price1'].shift(1)
        hf_analysis_df['prev_a1_v'] = hf_analysis_df['sell_volume1'].shift(1)
        hf_analysis_df['prev_b1_v'] = hf_analysis_df['buy_volume1'].shift(1)
        try:
            weighted_buy_vol = pd.Series(0, index=hf_analysis_df.index); weighted_sell_vol = pd.Series(0, index=hf_analysis_df.index)
            total_buy_value = pd.Series(0, index=hf_analysis_df.index); total_sell_value = pd.Series(0, index=hf_analysis_df.index)
            for i in range(1, 6):
                weight = 1 / i
                weighted_buy_vol += hf_analysis_df[f'buy_volume{i}'] * weight
                weighted_sell_vol += hf_analysis_df[f'sell_volume{i}'] * weight
                total_buy_value += hf_analysis_df[f'buy_volume{i}'] * hf_analysis_df[f'buy_price{i}']
                total_sell_value += hf_analysis_df[f'sell_volume{i}'] * hf_analysis_df[f'sell_price{i}']
            hf_analysis_df['imbalance'] = (weighted_buy_vol - weighted_sell_vol) / (weighted_buy_vol + weighted_sell_vol).replace(0, np.nan)
            hf_analysis_df['liquidity_supply_ratio'] = total_buy_value / total_sell_value.replace(0, np.nan)
        except Exception:
            hf_analysis_df['imbalance'] = np.nan
            hf_analysis_df['liquidity_supply_ratio'] = np.nan
        mf_trades = hf_analysis_df[is_main_force_trade].copy()
        if mf_trades.empty:
            return hf_analysis_df, features
        features['mf_trades'] = mf_trades
        buy_trades_mask = mf_trades['type'] == 'B'
        sell_trades_mask = mf_trades['type'] == 'S'
        features['buy_trades_mask'] = buy_trades_mask
        features['sell_trades_mask'] = sell_trades_mask
        total_mf_vol = mf_trades['volume'].sum()
        features['total_mf_vol'] = total_mf_vol
        mf_buy_trades = mf_trades[buy_trades_mask]
        mf_sell_trades = mf_trades[sell_trades_mask]
        if not mf_buy_trades.empty and mf_buy_trades['volume'].sum() > 0:
            features['hf_mf_buy_vwap'] = (mf_buy_trades['price'] * mf_buy_trades['volume']).sum() / mf_buy_trades['volume'].sum()
        if not mf_sell_trades.empty and mf_sell_trades['volume'].sum() > 0:
            features['hf_mf_sell_vwap'] = (mf_sell_trades['price'] * mf_sell_trades['volume']).sum() / mf_sell_trades['volume'].sum()
        if total_mf_vol > 0:
            features['mf_buy_vol'] = mf_buy_trades['volume'].sum()
            features['mf_sell_vol'] = mf_sell_trades['volume'].sum()
            offensive_buy_mask = (buy_trades_mask) & (mf_trades['price'] >= mf_trades['sell_price1'])
            offensive_sell_mask = (sell_trades_mask) & (mf_trades['price'] <= mf_trades['buy_price1'])
            offensive_volume = mf_trades[offensive_buy_mask | offensive_sell_mask]['volume'].sum()
            features['offensive_volume'] = offensive_volume
            features['passive_volume'] = total_mf_vol - offensive_volume
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
        results = {}
        if intraday_data.empty:
            return results
        raw_hf_df, common_data = self._prepare_behavioral_data(
            intraday_data, daily_data, tick_data, level5_data, realtime_data
        )
        hf_analysis_df, hf_features = self._engineer_hf_features(raw_hf_df, common_data.get('daily_total_volume', 0))
        current_date = daily_data.name.date()
        # 根据配置文件中的should_probe和probe_dates判断是否启用探针
        should_probe = self.debug_params.get('should_probe', False) and \
                       (current_date.strftime('%Y-%m-%d') in self.debug_params.get('probe_dates', []))
        context = {
            'intraday_data': intraday_data,
            'daily_data': daily_data, # 确保这里是原始的 daily_data Series
            'hf_analysis_df': hf_analysis_df,
            'common_data': common_data,
            'hf_features': hf_features,
            'main_force_net_flow_calibrated': main_force_net_flow_calibrated,
            'debug': {
                'should_probe': should_probe,
                'probe_dates': self.debug_params.get('probe_dates', []), # 更新为probe_dates
                'stock_code': stock_code
            }
        }
        if not hf_analysis_df.empty:
            results.update(AdvancedFundFlowMetricsService._calculate_main_force_profile_metrics(context))
            results.update(AdvancedFundFlowMetricsService._calculate_ofi_based_metrics(context))
            results.update(AdvancedFundFlowMetricsService._calculate_order_book_metrics(context))
            results.update(AdvancedFundFlowMetricsService._calculate_micro_dynamics_metrics(context))
            results.update(AdvancedFundFlowMetricsService._calculate_level5_order_flow_metrics(context))
        results.update(AdvancedFundFlowMetricsService._calculate_vwap_related_metrics(context))
        results.update(AdvancedFundFlowMetricsService._calculate_vwap_control_metrics(context))
        results.update(AdvancedFundFlowMetricsService._calculate_opening_battle_metrics(context))
        results.update(AdvancedFundFlowMetricsService._calculate_shadow_metrics(context))
        results.update(AdvancedFundFlowMetricsService._calculate_dip_rally_metrics(context))
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
    def _calculate_main_force_profile_metrics(context: dict) -> dict:
        """
        【V69.0 · 韧性升维版】(生产环境清洁版)
        - 核心逻辑: `main_force_conviction_index` 的“韧性”组件基于 `mid_price_delta < 0` (实际价格下跌) 计算，
                     以更精确地衡量主力在真实逆境中的托底决心。
        """
        hf_analysis_df = context['hf_analysis_df']
        common_data = context['common_data']
        hf_features = context['hf_features']
        hf_mf_buy_vwap = hf_features['hf_mf_buy_vwap']
        hf_mf_sell_vwap = hf_features['hf_mf_sell_vwap']
        import numpy as np
        metrics = {}
        atr = common_data['atr']
        daily_total_volume = common_data['daily_total_volume']
        mf_ofi_cumsum = hf_analysis_df['main_force_ofi'].cumsum().fillna(0)
        aggressiveness_component, trend_quality, closing_strength = 0.0, 0.0, 0.0
        if not mf_ofi_cumsum.empty and mf_ofi_cumsum.nunique() > 1:
            time_index = np.arange(len(mf_ofi_cumsum))
            trend_quality = np.corrcoef(time_index, mf_ofi_cumsum)[0, 1]
            trend_quality = np.nan_to_num(trend_quality)
            ofi_min, ofi_max = mf_ofi_cumsum.min(), mf_ofi_cumsum.max()
            closing_strength = (mf_ofi_cumsum.iloc[-1] - ofi_min) / (ofi_max - ofi_min) if (ofi_max - ofi_min) > 0 else 0.0
            closing_strength = np.nan_to_num(closing_strength)
            aggressiveness_component = trend_quality * closing_strength
        cost_tolerance_component = 0.0
        if pd.notna(hf_mf_buy_vwap) and pd.notna(hf_mf_sell_vwap) and pd.notna(atr) and atr > 0:
            cost_tolerance_component = (hf_mf_buy_vwap - hf_mf_sell_vwap) / atr
        price_pressure_zone = hf_analysis_df['mid_price_delta'] < 0
        mf_resilience_ofi = hf_analysis_df.loc[price_pressure_zone, 'main_force_ofi'].clip(lower=0).sum()
        total_mf_positive_ofi = hf_analysis_df['main_force_ofi'].clip(lower=0).sum()
        resilience_component = mf_resilience_ofi / total_mf_positive_ofi if total_mf_positive_ofi > 0 else 0.0
        metrics['main_force_conviction_index'] = (0.4 * aggressiveness_component + 0.4 * cost_tolerance_component + 0.2 * resilience_component) * 100
        mf_trades = hf_features['mf_trades']
        total_mf_vol = hf_features['total_mf_vol']
        if not mf_trades.empty and 'prev_mid_price' in mf_trades.columns:
            buy_trades_mask = hf_features['buy_trades_mask']
            sell_trades_mask = hf_features['sell_trades_mask']
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
                mf_buy_vol = hf_features['mf_buy_vol']
                mf_sell_vol = hf_features['mf_sell_vol']
                mf_total_activity_vol = mf_buy_vol + mf_sell_vol
                if mf_total_activity_vol > 0:
                    mf_net_vol = mf_buy_vol - mf_sell_vol
                    metrics['main_force_flow_directionality'] = (mf_net_vol / mf_total_activity_vol) * 100
        return metrics

    @staticmethod
    def _calculate_ofi_based_metrics(context: dict) -> dict:
        """
        【V72.2 · 主力订单流失衡聚合版】
        - 核心重构: `main_force_ofi` 和 `retail_ofi` (以及其买卖分量) 现在基于 `hf_analysis_df` 中
                    新定义的 `main_force_ofi` 和 `retail_ofi` 列（代表实际执行的净主动成交量）进行聚合计算，
                    并归一化为 [-1, 1] 的比率。
                    这确保了日度聚合指标反映的是主力/散户的实际成交行为。
        - 核心修复: `microstructure_efficiency_index` 现在使用 `hf_analysis_df['main_force_ofi']`
                    与 `mid_price_change` 进行相关性计算，以反映执行效率。
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
        if not hf_analysis_df.empty:
            # --- 计算主力订单流失衡比率 (基于实际执行成交量) ---
            mf_net_ofi_sum = hf_analysis_df['main_force_ofi'].sum()
            mf_abs_ofi_sum = hf_analysis_df['main_force_ofi'].abs().sum()
            mf_buy_ofi_sum = hf_analysis_df['main_force_ofi'].clip(lower=0).sum()
            mf_sell_ofi_sum = hf_analysis_df['main_force_ofi'].clip(upper=0).abs().sum() # 卖出量取绝对值
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
            mf_ofi_series = hf_analysis_df['main_force_ofi'] # 使用新的执行订单流
            price_change_series = hf_analysis_df['mid_price_change']
            if mf_ofi_series.var() > 0 and price_change_series.var() > 0:
                correlation = mf_ofi_series.corr(price_change_series)
                metrics['microstructure_efficiency_index'] = correlation
        return metrics

    @staticmethod
    def _calculate_level5_order_flow_metrics(context: dict) -> dict:
        """
        【V72.3 · Level5订单流失衡版】
        - 核心职责: 基于Level 5盘口数据，计算主力与散户的订单流失衡比率。
                    主要关注大额挂单所代表的潜在买卖压力。
        - 核心逻辑:
            1. 识别主力挂单：设定 Q_threshold，挂单量大于此阈值视为主力挂单。
            2. 统计主力/散户买卖盘压力：累加各档位的挂单量，并根据是否为主力挂单进行区分。
            3. 计算主力/散户订单流失衡比率：(买盘压力 - 卖盘压力) / (买盘压力 + 卖盘压力)。
            4. 结果归一化为 [-1, 1] 的比率。
        """
        hf_analysis_df = context['hf_analysis_df']
        common_data = context['common_data']
        metrics = {
            'main_force_level5_ofi': np.nan,
            'main_force_level5_buy_ofi': np.nan,
            'main_force_level5_sell_ofi': np.nan,
            'retail_level5_ofi': np.nan,
            'retail_level5_buy_ofi': np.nan,
            'retail_level5_sell_ofi': np.nan,
            'main_force_level5_ofi_dynamic': np.nan, # 动态变化分析
            'retail_level5_ofi_dynamic': np.nan,     # 动态变化分析
        }
        if hf_analysis_df.empty:
            return metrics
        # 设定主力挂单量阈值 (Q_threshold)
        # 可以根据 daily_total_volume 或 ATR 动态调整，这里先用一个固定值作为示例
        # 假设 1000 手（10万股）以上的挂单被认为是主力挂单
        Q_threshold = 1000 * 100 # 1000手 * 100股/手 = 10万股
        # 权重可以根据档位远近设置，这里简化为等权重
        weights = [1.0, 0.8, 0.6, 0.4, 0.2] # 越靠近盘口权重越高
        # 存储每个快照的 Level 5 OFI
        main_force_ofi_snapshots = []
        retail_ofi_snapshots = []
        # 遍历每个快照
        for _, row in hf_analysis_df.iterrows():
            main_force_bid_pressure = 0.0
            main_force_ask_pressure = 0.0
            retail_bid_pressure = 0.0
            retail_ask_pressure = 0.0
            for i in range(1, 6):
                buy_vol_col = f'buy_volume{i}'
                buy_price_col = f'buy_price{i}'
                sell_vol_col = f'sell_volume{i}'
                sell_price_col = f'sell_price{i}'
                buy_vol = row.get(buy_vol_col, 0)
                sell_vol = row.get(sell_vol_col, 0)
                # 确保价格有效，避免0价格导致的问题
                buy_price = row.get(buy_price_col, 0)
                sell_price = row.get(sell_price_col, 0)
                if buy_vol > 0 and buy_price > 0:
                    if buy_vol >= Q_threshold:
                        main_force_bid_pressure += buy_vol * weights[i-1]
                    else:
                        retail_bid_pressure += buy_vol * weights[i-1]
                if sell_vol > 0 and sell_price > 0:
                    if sell_vol >= Q_threshold:
                        main_force_ask_pressure += sell_vol * weights[i-1]
                    else:
                        retail_ask_pressure += sell_vol * weights[i-1]
            # 计算当前快照的主力订单流失衡比率
            total_mf_pressure = main_force_bid_pressure + main_force_ask_pressure
            if total_mf_pressure > 0:
                main_force_ofi_snapshots.append((main_force_bid_pressure - main_force_ask_pressure) / total_mf_pressure)
            else:
                main_force_ofi_snapshots.append(0.0) # 无主力挂单时，失衡为0
            # 计算当前快照的散户订单流失衡比率
            total_retail_pressure = retail_bid_pressure + retail_ask_pressure
            if total_retail_pressure > 0:
                retail_ofi_snapshots.append((retail_bid_pressure - retail_ask_pressure) / total_retail_pressure)
            else:
                retail_ofi_snapshots.append(0.0) # 无散户挂单时，失衡为0
        # 将快照结果转换为 Series
        main_force_ofi_series = pd.Series(main_force_ofi_snapshots, index=hf_analysis_df.index)
        retail_ofi_series = pd.Series(retail_ofi_snapshots, index=hf_analysis_df.index)
        # 对整个交易日的主力/散户 Level 5 OFI 进行加权平均
        # 可以使用时间差作为权重，或者简单平均
        time_diffs = hf_analysis_df.index.to_series().diff().dt.total_seconds().fillna(0)
        total_time = time_diffs.sum()
        if total_time > 0:
            # 主力 Level 5 OFI
            metrics['main_force_level5_ofi'] = np.average(main_force_ofi_series.dropna(), weights=time_diffs[main_force_ofi_series.notna()])
            metrics['main_force_level5_buy_ofi'] = np.average(main_force_ofi_series.clip(lower=0).dropna(), weights=time_diffs[main_force_ofi_series.notna()])
            metrics['main_force_level5_sell_ofi'] = np.average(main_force_ofi_series.clip(upper=0).dropna(), weights=time_diffs[main_force_ofi_series.notna()])
            # 散户 Level 5 OFI
            metrics['retail_level5_ofi'] = np.average(retail_ofi_series.dropna(), weights=time_diffs[retail_ofi_series.notna()])
            metrics['retail_level5_buy_ofi'] = np.average(retail_ofi_series.clip(lower=0).dropna(), weights=time_diffs[retail_ofi_series.notna()])
            metrics['retail_level5_sell_ofi'] = np.average(retail_ofi_series.clip(upper=0).dropna(), weights=time_diffs[retail_ofi_series.notna()])
            # 动态变化分析：计算日内 Level 5 OFI 的平均变化率
            metrics['main_force_level5_ofi_dynamic'] = main_force_ofi_series.diff().mean()
            metrics['retail_level5_ofi_dynamic'] = retail_ofi_series.diff().mean()
        else:
            # 如果没有有效时间差，则直接取平均值（或保持NaN）
            metrics['main_force_level5_ofi'] = main_force_ofi_series.mean()
            metrics['main_force_level5_buy_ofi'] = main_force_ofi_series.clip(lower=0).mean()
            metrics['main_force_level5_sell_ofi'] = main_force_ofi_series.clip(upper=0).mean()
            metrics['retail_level5_ofi'] = retail_ofi_series.mean()
            metrics['retail_level5_buy_ofi'] = retail_ofi_series.clip(lower=0).mean()
            metrics['retail_level5_sell_ofi'] = retail_ofi_series.clip(upper=0).mean()
            metrics['main_force_level5_ofi_dynamic'] = main_force_ofi_series.diff().mean()
            metrics['retail_level5_ofi_dynamic'] = retail_ofi_series.diff().mean()
        return metrics

    @staticmethod
    def _calculate_order_book_metrics(context: dict) -> dict:
        """
        【V71.0 · 终极生产版】(生产环境清洁版)
        【V72.0 · 资金流拆分版】
        - 核心增强: 拆分 `order_book_clearing_rate` 和 `order_book_imbalance` 为买卖双方贡献。
        """
        hf_analysis_df = context['hf_analysis_df']
        common_data = context['common_data']
        import numpy as np
        metrics = {}
        daily_total_volume = common_data['daily_total_volume']
        large_orders_df = hf_analysis_df[hf_analysis_df['amount'] > 200000]
        if not large_orders_df.empty:
            metrics['observed_large_order_size_avg'] = large_orders_df['amount'].mean()
        up_ticks = hf_analysis_df[hf_analysis_df['mid_price_change'] > 0]
        down_ticks = hf_analysis_df[hf_analysis_df['mid_price_change'] < 0]
        if not up_ticks.empty and not down_ticks.empty and up_ticks['mid_price_change'].sum() > 0 and down_ticks['mid_price_change'].abs().sum() > 0:
            vol_per_tick_up = up_ticks['volume'].sum() / (up_ticks['mid_price_change'].sum() * 100)
            vol_per_tick_down = down_ticks['volume'].sum() / (down_ticks['mid_price_change'].abs().sum() * 100)
            if vol_per_tick_down > 1e-9: # 避免除以零
                asymmetry_ratio = vol_per_tick_up / vol_per_tick_down
                # MODIFIED BLOCK START
                # 修正 micro_price_impact_asymmetry 的计算逻辑，使其可以为负值
                # np.log(ratio) 会在 ratio < 1 时为负，ratio > 1 时为正
                metrics['micro_price_impact_asymmetry'] = np.log(asymmetry_ratio) if asymmetry_ratio > 1e-9 else np.nan
                # MODIFIED BLOCK END
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
                metrics['order_book_imbalance'] = np.average(hf_analysis_df['imbalance'].dropna(), weights=time_diffs[hf_analysis_df['imbalance'].notna()]) * 100
                metrics['order_book_liquidity_supply'] = np.average(hf_analysis_df['liquidity_supply_ratio'].dropna(), weights=time_diffs[hf_analysis_df['liquidity_supply_ratio'].notna()])
                bid_liquidity_cols = [f'buy_volume{i}' for i in range(1, 6)]
                ask_liquidity_cols = [f'sell_volume{i}' for i in range(1, 6)]
                bid_depth_series = hf_analysis_df[bid_liquidity_cols].sum(axis=1)
                ask_depth_series = hf_analysis_df[ask_liquidity_cols].sum(axis=1)
                metrics['bid_side_liquidity'] = np.average(bid_depth_series.dropna(), weights=time_diffs[bid_depth_series.notna()]) if bid_depth_series.notna().any() else np.nan
                metrics['ask_side_liquidity'] = np.average(ask_depth_series.dropna(), weights=time_diffs[ask_depth_series.notna()]) if ask_depth_series.notna().any() else np.nan
            if 'market_vol_delta' in hf_analysis_df.columns and hf_analysis_df['imbalance'].var() > 1e-9 and hf_analysis_df['market_vol_delta'].var() > 1e-9:
                correlation_value = hf_analysis_df['imbalance'].corr(hf_analysis_df['market_vol_delta'])
                metrics['imbalance_effectiveness'] = correlation_value
        except Exception:
            pass
        try:
            df_static = hf_analysis_df.copy()
            large_order_threshold_value = 500000
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
        except Exception:
            metrics['large_order_pressure'] = np.nan; metrics['large_order_support'] = np.nan
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
        【V71.0 · 终极生产版】(生产环境清洁版)
        【V72.0 · 资金流拆分版】
        - 核心增强: 拆分 `opening_battle_result` 为买卖双方强度。
        """
        intraday_data = context['intraday_data']
        hf_analysis_df = context['hf_analysis_df']
        common_data = context['common_data']
        from datetime import time
        import numpy as np
        metrics = {}
        atr = common_data['atr']
        opening_battle_df = intraday_data[(intraday_data.index.time >= time(9, 30)) & (intraday_data.index.time <= time(9, 45))]
        if not opening_battle_df.empty and len(opening_battle_df) > 1 and pd.notna(atr) and atr > 0:
            if not hf_analysis_df.empty:
                opening_hf_df = hf_analysis_df[(hf_analysis_df.index.time >= time(9, 30)) & (hf_analysis_df.index.time <= time(9, 45))]
                if not opening_hf_df.empty:
                    price_gain_hf = (opening_hf_df['price'].iloc[-1] - opening_hf_df['price'].iloc[0]) / atr
                    mf_ofi_opening = opening_hf_df['main_force_ofi'].sum()
                    total_abs_ofi_opening = opening_hf_df['ofi'].abs().sum()
                    mf_ofi_dominance = mf_ofi_opening / total_abs_ofi_opening if total_abs_ofi_opening > 0 else 0
                    metrics['opening_battle_result'] = price_gain_hf * (1 + mf_ofi_dominance) * 100
                    # 新增拆分指标
                    mf_buy_ofi_opening = opening_hf_df['main_force_ofi'].clip(lower=0).sum() # 新增行
                    mf_sell_ofi_opening = opening_hf_df['main_force_ofi'].clip(upper=0).sum() # 新增行
                    metrics['opening_buy_strength'] = (mf_buy_ofi_opening / total_abs_ofi_opening) * 100 if total_abs_ofi_opening > 0 else np.nan # 新增行
                    metrics['opening_sell_strength'] = (abs(mf_sell_ofi_opening) / total_abs_ofi_opening) * 100 if total_abs_ofi_opening > 0 else np.nan # 新增行
            else:
                if 'close' in opening_battle_df.columns and 'open' in opening_battle_df.columns and 'vol_shares' in opening_battle_df.columns and 'minute_vwap' in opening_battle_df.columns and 'main_force_net_vol' in opening_battle_df.columns:
                    price_gain = (opening_battle_df['close'].iloc[-1] - opening_battle_df['open'].iloc[0]) / atr
                    battle_amount = (opening_battle_df['vol_shares'] * opening_battle_df['minute_vwap']).sum()
                    if battle_amount > 0:
                        mf_power = opening_battle_df['main_force_net_vol'].sum() * opening_battle_df['minute_vwap'].mean() / battle_amount
                        metrics['opening_battle_result'] = np.sign(price_gain) * np.sqrt(abs(price_gain)) * (1 + mf_power) * 100
                        # Fallback for split metrics
                        mf_buy_vol_opening = opening_battle_df['main_force_buy_vol'].sum() # 新增行
                        mf_sell_vol_opening = opening_battle_df['main_force_sell_vol'].sum() # 新增行
                        total_vol_opening = opening_battle_df['vol_shares'].sum() # 新增行
                        metrics['opening_buy_strength'] = (mf_buy_vol_opening / total_vol_opening) * 100 if total_vol_opening > 0 else np.nan # 新增行
                        metrics['opening_sell_strength'] = (mf_sell_vol_opening / total_vol_opening) * 100 if total_vol_opening > 0 else np.nan # 新增行
        return metrics

    @staticmethod
    def _calculate_shadow_metrics(context: dict) -> dict:
        hf_analysis_df = context['hf_analysis_df']
        common_data = context['common_data']
        hf_features = context['hf_features']
        should_probe = context['debug']['should_probe']
        stock_code = context['debug']['stock_code']
        current_date = context['daily_data'].name.date()
        import numpy as np
        metrics = {}
        day_open, day_close = common_data['day_open'], common_data['day_close']
        day_high, day_low = common_data['day_high'], common_data['day_low']
        atr = common_data.get('atr', 0)
        daily_total_amount = common_data.get('daily_total_amount', 0)
        if pd.notna(day_open) and pd.notna(day_close) and pd.notna(day_high) and pd.notna(day_low) and pd.notna(atr) and atr > 0 and daily_total_amount > 0:
            day_range = day_high - day_low
            if day_range <= 0:
                return metrics
            market_value_efficiency = (day_range / atr) / (daily_total_amount / 10000)
            body_high, body_low = max(day_open, day_close), min(day_open, day_close)
            if day_low < body_low:
                price_recovery_norm = (body_low - day_low) / atr
                if not hf_analysis_df.empty:
                    hf_shadow_zone = hf_analysis_df[hf_analysis_df['price'] < body_low]
                    mf_trades_in_shadow = hf_features['mf_trades'].loc[hf_features['mf_trades'].index.intersection(hf_shadow_zone.index)]
                    if not mf_trades_in_shadow.empty:
                        mf_buy_amount = mf_trades_in_shadow[mf_trades_in_shadow['type'] == 'B']['amount'].sum()
                        mf_sell_amount = mf_trades_in_shadow[mf_trades_in_shadow['type'] == 'S']['amount'].sum()
                        mf_net_buy_amount_10k = (mf_buy_amount - mf_sell_amount) / 10000
                        if mf_net_buy_amount_10k > 0 and market_value_efficiency > 0:
                            absorption_efficiency = price_recovery_norm / mf_net_buy_amount_10k
                            normalized_strength = absorption_efficiency / market_value_efficiency
                            compressed_strength = np.log1p(normalized_strength)
                            metrics['lower_shadow_absorption_strength'] = np.tanh(compressed_strength) * 100
            if day_high > body_high:
                price_rejection_norm = (day_high - body_high) / atr
                if not hf_analysis_df.empty:
                    hf_shadow_zone = hf_analysis_df[hf_analysis_df['price'] > body_high]
                    mf_trades_in_shadow = hf_features['mf_trades'].loc[hf_features['mf_trades'].index.intersection(hf_shadow_zone.index)]
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
    def _calculate_dip_rally_metrics(context: dict) -> dict:
        """
        【V72.5 · 时区兼容修复版】
        - 核心修复: 解决 `TypeError: Cannot compare tz-naive and tz-aware timestamps`。
                    在对 `mf_trades.index.values` (被视为时区-naive的 `datetime64[ns]` 数组)
                    与 `start_time` 和 `end_time` (时区-aware的 `pd.Timestamp` 对象) 进行比较时，
                    将 `start_time` 和 `end_time` 显式转换为时区-naive，以确保比较操作的时区一致性。
        """
        intraday_data = context['intraday_data']
        hf_analysis_df = context['hf_analysis_df']
        common_data = context['common_data']
        hf_features = context['hf_features']
        should_probe = context['debug']['should_probe']
        stock_code = context['debug']['stock_code']
        current_date = context['daily_data'].name.date()
        from scipy.signal import find_peaks
        from datetime import time
        import numpy as np
        import pandas as pd # 确保 pandas 已导入，以便使用 Timestamp 的方法
        metrics = {}
        daily_vwap = common_data['daily_vwap']
        atr = common_data['atr']
        daily_total_amount = common_data.get('daily_total_amount', 0)
        day_high, day_low = common_data['day_high'], common_data['day_low']
        continuous_trading_df = intraday_data[intraday_data.index.time < time(14, 57)].copy()
        if not continuous_trading_df.empty and 'minute_vwap' in continuous_trading_df.columns:
            peaks, _ = find_peaks(continuous_trading_df['minute_vwap'].values)
            troughs, _ = find_peaks(-continuous_trading_df['minute_vwap'].values)
            turning_points = sorted(list(set(np.concatenate(([0], troughs, peaks, [len(continuous_trading_df)-1])))))
            if not hf_analysis_df.empty and pd.notna(daily_vwap) and pd.notna(atr) and atr > 0:
                absorption_zone_hf = hf_analysis_df[hf_analysis_df['price'] < daily_vwap]
                if not absorption_zone_hf.empty:
                    mf_net_buy_vol = absorption_zone_hf['main_force_ofi'].clip(lower=0).sum()
                    mf_net_sell_vol = absorption_zone_hf['main_force_ofi'].clip(upper=0).sum()
                    price_drop_vs_vwap = (daily_vwap - absorption_zone_hf['price']).clip(lower=0)
                    price_weighted_effort = (price_drop_vs_vwap * absorption_zone_hf['volume']).sum()
                    if price_weighted_effort > 0:
                        absorption_efficiency = mf_net_buy_vol / price_weighted_effort
                        metrics['dip_absorption_power'] = np.tanh(absorption_efficiency * atr) * 100
                        metrics['dip_buy_absorption_strength'] = np.tanh(mf_net_buy_vol / price_weighted_effort * atr) * 100
                        if mf_net_sell_vol < 0:
                            resistance_efficiency = abs(mf_net_sell_vol) / price_weighted_effort
                            metrics['dip_sell_pressure_resistance'] = np.tanh(resistance_efficiency * atr) * 100
                total_mf_net_sell_amount_in_rallies = 0
                total_mf_net_buy_amount_in_rallies = 0
                total_rally_price_change_norm = 0
                mf_trades = hf_features['mf_trades']
                for i in range(len(turning_points) - 1):
                    start_idx, end_idx = turning_points[i], turning_points[i+1]
                    window_df = continuous_trading_df.iloc[start_idx:end_idx+1]
                    if window_df.empty or len(window_df) < 2: continue
                    if window_df['minute_vwap'].iloc[-1] > window_df['minute_vwap'].iloc[0]:
                        start_time, end_time = window_df.index[0], window_df.index[-1]
                        # 核心修复：将 start_time 和 end_time 转换为时区-naive
                        start_time_naive = start_time.tz_localize(None) if start_time.tz is not None else start_time
                        end_time_naive = end_time.tz_localize(None) if end_time.tz is not None else end_time
                        # 使用时区-naive的时间戳进行比较
                        mf_trades_in_rally = mf_trades[(mf_trades.index.values >= start_time_naive) & (mf_trades.index.values <= end_time_naive)]
                        if not mf_trades_in_rally.empty:
                            mf_buy_amount = mf_trades_in_rally[mf_trades_in_rally['type'] == 'B']['amount'].sum()
                            mf_sell_amount = mf_trades_in_rally[mf_trades_in_rally['type'] == 'S']['amount'].sum()
                            total_mf_net_sell_amount_in_rallies += (mf_sell_amount - mf_buy_amount)
                            total_mf_net_buy_amount_in_rallies += (mf_buy_amount - mf_sell_amount)
                            price_change_in_rally = window_df['minute_vwap'].iloc[-1] - window_df['minute_vwap'].iloc[0]
                            total_rally_price_change_norm += (price_change_in_rally / atr)
                day_range = day_high - day_low
                if total_rally_price_change_norm > 0 and day_range > 0 and daily_total_amount > 0:
                    deception_coeff = (total_mf_net_sell_amount_in_rallies / 10000) / total_rally_price_change_norm
                    market_price_cost = (daily_total_amount / 10000) / (day_range / atr)
                    if market_price_cost > 0:
                        normalized_pressure = deception_coeff / market_price_cost
                        metrics['rally_distribution_pressure'] = np.tanh(normalized_pressure) * 100
                        metrics['rally_sell_distribution_intensity'] = np.tanh(deception_coeff / market_price_cost) * 100
                        if total_mf_net_buy_amount_in_rallies < 0: # 只有当主力净买入为负时才计算支撑弱点
                            weakness_coeff = (abs(total_mf_net_buy_amount_in_rallies) / 10000) / total_rally_price_change_norm
                            metrics['rally_buy_support_weakness'] = np.tanh(weakness_coeff / market_price_cost) * 100
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
        """
        【V71.0 · 终极生产版】(生产环境清洁版)
        【V72.0 · 资金流拆分版】
        - 核心增强: 拆分 `pre_closing_posturing` 和 `closing_auction_ambush` 为买卖双方姿态/伏击。
        """
        intraday_data = context['intraday_data']
        hf_analysis_df = context['hf_analysis_df']
        common_data = context['common_data']
        from datetime import time
        import numpy as np
        metrics = {}
        day_close = common_data['day_close']
        daily_vwap = common_data['daily_vwap']
        atr = common_data['atr']
        continuous_trading_df = intraday_data[intraday_data.index.time < time(14, 57)].copy()
        if not continuous_trading_df.empty and pd.notna(atr) and atr > 0:
            auction_df = intraday_data[intraday_data.index.time >= time(14, 57)]
            if not auction_df.empty:
                avg_minute_vol = continuous_trading_df['vol_shares'].mean()
                auction_vol = auction_df['vol_shares'].sum()
                VolumeAnomaly = np.log1p((auction_vol / 3) / avg_minute_vol) if avg_minute_vol > 0 else 0.0
                if not hf_analysis_df.empty:
                    pre_auction_df = hf_analysis_df[hf_analysis_df.index.time < time(14, 57)]
                    if not pre_auction_df.empty:
                        pre_auction_snapshot = pre_auction_df.iloc[-1]
                        pre_auction_mid = pre_auction_snapshot['mid_price']
                        pre_auction_imbalance = pre_auction_snapshot['imbalance']
                        PriceDeviation = (day_close - pre_auction_mid) / atr if pd.notna(pre_auction_mid) else 0.0
                        Deception = -np.sign(PriceDeviation) * pre_auction_imbalance if pd.notna(pre_auction_imbalance) else 0.0
                        metrics['closing_auction_ambush'] = PriceDeviation * VolumeAnomaly * (1 + Deception) * 100
                        # 新增拆分指标
                        mf_auction_buy_vol = hf_analysis_df[(hf_analysis_df.index.time >= time(14, 57)) & (hf_analysis_df['amount'] > 200000) & (hf_analysis_df['type'] == 'B')]['volume'].sum() # 新增行
                        mf_auction_sell_vol = hf_analysis_df[(hf_analysis_df.index.time >= time(14, 57)) & (hf_analysis_df['amount'] > 200000) & (hf_analysis_df['type'] == 'S')]['volume'].sum() # 新增行
                        total_auction_vol = hf_analysis_df[hf_analysis_df.index.time >= time(14, 57)]['volume'].sum() # 新增行
                        if total_auction_vol > 0: # 新增行
                            metrics['closing_auction_buy_ambush'] = (mf_auction_buy_vol / total_auction_vol) * PriceDeviation * VolumeAnomaly * 100 # 新增行
                            metrics['closing_auction_sell_ambush'] = (mf_auction_sell_vol / total_auction_vol) * PriceDeviation * VolumeAnomaly * 100 # 新增行
                else:
                    pre_auction_close = continuous_trading_df['close'].iloc[-1]
                    PriceImpact = (day_close - pre_auction_close) / atr if pd.notna(pre_auction_close) else 0.0
                    metrics['closing_auction_ambush'] = PriceImpact * VolumeAnomaly * 100
                    # Fallback for split metrics
                    mf_auction_buy_vol_fallback = auction_df['main_force_buy_vol'].sum() # 新增行
                    mf_auction_sell_vol_fallback = auction_df['main_force_sell_vol'].sum() # 新增行
                    total_auction_vol_fallback = auction_df['vol_shares'].sum() # 新增行
                    if total_auction_vol_fallback > 0: # 新增行
                        metrics['closing_auction_buy_ambush'] = (mf_auction_buy_vol_fallback / total_auction_vol_fallback) * PriceImpact * VolumeAnomaly * 100 # 新增行
                        metrics['closing_auction_sell_ambush'] = (mf_auction_sell_vol_fallback / total_auction_vol_fallback) * PriceImpact * VolumeAnomaly * 100 # 新增行
            posturing_df = continuous_trading_df[continuous_trading_df.index.time >= time(14, 30)]
            if pd.notna(daily_vwap) and not posturing_df.empty:
                if not hf_analysis_df.empty:
                    posturing_hf_df = hf_analysis_df[hf_analysis_df.index.time >= time(14, 30)]
                    if not posturing_hf_df.empty:
                        time_diffs = posturing_hf_df.index.to_series().diff().dt.total_seconds().fillna(0)
                        if time_diffs.sum() > 0:
                            avg_imbalance = np.average(posturing_hf_df['imbalance'].dropna(), weights=time_diffs[posturing_hf_df['imbalance'].notna()])
                            avg_spread = (posturing_hf_df['sell_price1'] - posturing_hf_df['buy_price1']).mean()
                            normalized_imbalance = avg_imbalance * (avg_spread / atr) if pd.notna(avg_spread) and avg_spread > 0 else 0
                            metrics['pre_closing_posturing'] = normalized_imbalance * 100
                            # 新增拆分指标
                            mf_buy_ofi_posturing = posturing_hf_df['main_force_ofi'].clip(lower=0).sum() # 新增行
                            mf_sell_ofi_posturing = posturing_hf_df['main_force_ofi'].clip(upper=0).sum() # 新增行
                            total_mf_ofi_abs_posturing = posturing_hf_df['main_force_ofi'].abs().sum() # 新增行
                            if total_mf_ofi_abs_posturing > 0: # 新增行
                                metrics['pre_closing_buy_posture'] = (mf_buy_ofi_posturing / total_mf_ofi_abs_posturing) * normalized_imbalance * 100 # 新增行
                                metrics['pre_closing_sell_posture'] = (abs(mf_sell_ofi_posturing) / total_mf_ofi_abs_posturing) * normalized_imbalance * 100 # 新增行
                else:
                    if 'vol_shares' in posturing_df.columns and 'minute_vwap' in posturing_df.columns and 'main_force_net_vol' in posturing_df.columns:
                        posturing_vwap = (posturing_df['vol_shares'] * posturing_df['minute_vwap']).sum() / posturing_df['vol_shares'].sum()
                        price_posture = (posturing_vwap - daily_vwap) / atr
                        posturing_amount = (posturing_df['vol_shares'] * posturing_df['minute_vwap']).sum()
                        if posturing_amount > 0:
                            force_posture = (posturing_df['main_force_net_vol'].sum() * posturing_vwap) / posturing_amount
                            metrics['pre_closing_posturing'] = (0.6 * price_posture + 0.4 * force_posture) * 100
                            # Fallback for split metrics
                            mf_buy_vol_posturing = posturing_df['main_force_buy_vol'].sum() # 新增行
                            mf_sell_vol_posturing = posturing_df['main_force_sell_vol'].sum() # 新增行
                            total_mf_vol_posturing = posturing_df['main_force_buy_vol'].sum() + posturing_df['main_force_sell_vol'].sum() # 新增行
                            if total_mf_vol_posturing > 0: # 新增行
                                metrics['pre_closing_buy_posture'] = (mf_buy_vol_posturing / total_mf_vol_posturing) * (0.6 * price_posture + 0.4 * force_posture) * 100 # 新增行
                                metrics['pre_closing_sell_posture'] = (mf_sell_vol_posturing / total_mf_vol_posturing) * (0.6 * price_posture + 0.4 * force_posture) * 100 # 新增行
        return metrics

    @staticmethod
    def _calculate_hidden_accumulation_metrics(context: dict) -> dict:
        """
        【V71.0 · 终极生产版】(生产环境清洁版)
        """
        intraday_data = context['intraday_data']
        hf_analysis_df = context['hf_analysis_df']
        common_data = context['common_data']
        import numpy as np
        metrics = {}
        daily_vwap = common_data['daily_vwap']
        if not hf_analysis_df.empty and pd.notna(daily_vwap):
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
        else:
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
        【V71.0 · 终极生产版】(生产环境清洁版)
        【V72.0 · 资金流拆分版】
        - 核心增强: 拆分 `vwap_control_strength` 为买卖双方控制强度。
        """
        intraday_data = context['intraday_data']
        hf_analysis_df = context['hf_analysis_df']
        common_data = context['common_data']
        import numpy as np
        import pandas as pd
        metrics = {
            'vwap_control_strength': np.nan,
            'vwap_buy_control_strength': np.nan, # 新增行
            'vwap_sell_control_strength': np.nan, # 新增行
        }
        daily_vwap = common_data['daily_vwap']
        daily_total_volume = common_data['daily_total_volume']
        atr = common_data['atr']
        if pd.isna(daily_vwap) or pd.isna(daily_total_volume) or daily_total_volume <= 0 or pd.isna(atr) or atr <= 0:
            return metrics
        if not hf_analysis_df.empty and 'ofi' in hf_analysis_df.columns and 'main_force_ofi' in hf_analysis_df.columns:
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
                # 新增拆分指标
                mf_buy_ofi_in_zone = zone_hf_df['main_force_ofi'].clip(lower=0).sum() # 新增行
                mf_sell_ofi_in_zone = zone_hf_df['main_force_ofi'].clip(upper=0).sum() # 新增行
                total_mf_ofi_in_zone = zone_hf_df['main_force_ofi'].abs().sum() # 新增行
                if total_mf_ofi_in_zone > 0: # 新增行
                    metrics['vwap_buy_control_strength'] = (mf_buy_ofi_in_zone / total_mf_ofi_in_zone) * absorption_ratio * volume_significance * 100 # 新增行
                    metrics['vwap_sell_control_strength'] = (abs(mf_sell_ofi_in_zone) / total_mf_ofi_in_zone) * absorption_ratio * volume_significance * 100 # 新增行
        else:
            if 'minute_vwap' in intraday_data.columns and 'vol_shares' in intraday_data.columns:
                price_deviation_value = (intraday_data['minute_vwap'] - daily_vwap) * intraday_data['vol_shares']
                metrics['vwap_control_strength'] = price_deviation_value.sum() / (atr * daily_total_volume)
                # Fallback for split metrics
                if 'main_force_buy_vol' in intraday_data.columns and 'main_force_sell_vol' in intraday_data.columns: # 新增行
                    mf_buy_vol_in_zone = intraday_data['main_force_buy_vol'].sum() # 新增行
                    mf_sell_vol_in_zone = intraday_data['main_force_sell_vol'].sum() # 新增行
                    total_mf_vol_in_zone = mf_buy_vol_in_zone + mf_sell_vol_in_zone # 新增行
                    if total_mf_vol_in_zone > 0: # 新增行
                        metrics['vwap_buy_control_strength'] = (mf_buy_vol_in_zone / total_mf_vol_in_zone) * metrics['vwap_control_strength'] # 新增行
                        metrics['vwap_sell_control_strength'] = (mf_sell_vol_in_zone / total_mf_vol_in_zone) * metrics['vwap_control_strength'] # 新增行
        return metrics

    @staticmethod
    def _calculate_cmf_metrics(context: dict) -> dict:
        """
        【V70.0 · 背离放大器终版】(生产环境清洁版)
        - 核心逻辑: 引入“背离放大器”，当主力CMF与市场CMF异号时，加权突显最关键的“方向背离”信号，
                     使指标能更敏锐地捕捉市场核心矛盾。
        """
        intraday_data = context['intraday_data']
        hf_analysis_df = context['hf_analysis_df']
        import numpy as np
        import pandas as pd
        metrics = {}
        if not hf_analysis_df.empty and 'price' in hf_analysis_df.columns and 'main_force_ofi' in hf_analysis_df.columns:
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
        else:
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
        【V71.0 · 终极生产版】(生产环境清洁版)
        【V72.0 · 资金流拆分版】
        - 核心增强: 拆分 `main_force_on_peak_flow` 为买卖双方在主峰区的资金流。
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
            'main_force_on_peak_buy_flow': np.nan, # 新增行
            'main_force_on_peak_sell_flow': np.nan, # 新增行
        }
        daily_total_amount = common_data['daily_total_amount']
        def _calculate_vpoc_from_ticks(df: pd.DataFrame, volume_col: str, price_col: str, bins: int = 50) -> tuple[float, pd.Interval]:
            if df.empty or df[price_col].nunique() < 2:
                return np.nan, None
            price_bins = pd.cut(df[price_col], bins=bins, duplicates='drop')
            vol_profile = df.groupby(price_bins)[volume_col].sum()
            if vol_profile.empty:
                return np.nan, None
            vpoc_interval = vol_profile.idxmax()
            return vpoc_interval.mid, vpoc_interval
        if not hf_analysis_df.empty:
            global_vpoc_price, global_vpoc_interval = _calculate_vpoc_from_ticks(hf_analysis_df, 'volume', 'price')
            mf_trades = hf_features['mf_trades']
            mf_vpoc, _ = _calculate_vpoc_from_ticks(mf_trades, 'volume', 'price')
            metrics['main_force_vpoc'] = mf_vpoc
            if pd.notna(global_vpoc_price) and global_vpoc_price > 0 and pd.notna(mf_vpoc):
                metrics['mf_vpoc_premium'] = (mf_vpoc / global_vpoc_price - 1) * 100
            if global_vpoc_interval is not None:
                peak_zone_mf_trades = mf_trades[
                    (mf_trades['price'] >= global_vpoc_interval.left) &
                    (mf_trades['price'] < global_vpoc_interval.right)
                ]
                if not peak_zone_mf_trades.empty:
                    net_amount_on_peak = np.where(
                        peak_zone_mf_trades['type'] == 'B',
                        peak_zone_mf_trades['amount'],
                        -peak_zone_mf_trades['amount']
                    ).sum()
                    if daily_total_amount > 0:
                        metrics['main_force_on_peak_flow'] = np.tanh(net_amount_on_peak / daily_total_amount)
                    # 新增拆分指标
                    mf_buy_amount_on_peak = peak_zone_mf_trades[peak_zone_mf_trades['type'] == 'B']['amount'].sum() # 新增行
                    mf_sell_amount_on_peak = peak_zone_mf_trades[peak_zone_mf_trades['type'] == 'S']['amount'].sum() # 新增行
                    if daily_total_amount > 0: # 新增行
                        metrics['main_force_on_peak_buy_flow'] = np.tanh(mf_buy_amount_on_peak / daily_total_amount) # 新增行
                        metrics['main_force_on_peak_sell_flow'] = np.tanh(mf_sell_amount_on_peak / daily_total_amount) # 新增行
        else:
            if 'main_force_net_vol' in intraday_data.columns and 'minute_vwap' in intraday_data.columns and 'vol_shares' in intraday_data.columns:
                vp_global = intraday_data.groupby(pd.cut(intraday_data['minute_vwap'], bins=30, duplicates='drop'))['vol_shares'].sum()
                global_vpoc_price = np.nan
                if not vp_global.empty:
                    vpoc_interval = vp_global.idxmax()
                    global_vpoc_price = vpoc_interval.mid
                    peak_zone_df = intraday_data[(intraday_data['minute_vwap'] >= vpoc_interval.left) & (intraday_data['minute_vwap'] < vpoc_interval.right)]
                    if not peak_zone_df.empty:
                        mf_net_vol_on_peak = peak_zone_df['main_force_net_vol'].sum()
                        if daily_total_amount > 0:
                            normalized_mf_on_peak_flow = np.tanh((mf_net_vol_on_peak * global_vpoc_price) / daily_total_amount)
                            metrics['main_force_on_peak_flow'] = normalized_mf_on_peak_flow
                        # Fallback for split metrics
                        mf_buy_vol_on_peak_fallback = peak_zone_df['main_force_buy_vol'].sum() # 新增行
                        mf_sell_vol_on_peak_fallback = peak_zone_df['main_force_sell_vol'].sum() # 新增行
                        if daily_total_amount > 0: # 新增行
                            metrics['main_force_on_peak_buy_flow'] = np.tanh((mf_buy_vol_on_peak_fallback * global_vpoc_price) / daily_total_amount) # 新增行
                            metrics['main_force_on_peak_sell_flow'] = np.tanh((mf_sell_vol_on_peak_fallback * global_vpoc_price) / daily_total_amount) # 新增行
                mf_net_buy_df = intraday_data[intraday_data['main_force_net_vol'] > 0]
                if not mf_net_buy_df.empty:
                    vp_mf = mf_net_buy_df.groupby(pd.cut(mf_net_buy_df['minute_vwap'], bins=30, duplicates='drop'))['main_force_net_vol'].sum()
                    if not vp_mf.empty:
                        mf_vpoc = vp_mf.idxmax().mid
                        metrics['main_force_vpoc'] = mf_vpoc
                        if pd.notna(global_vpoc_price) and global_vpoc_price > 0 and pd.notna(mf_vpoc):
                            metrics['mf_vpoc_premium'] = (mf_vpoc / global_vpoc_price - 1) * 100
        return metrics

    @staticmethod
    def _calculate_liquidity_swap_metrics(context: dict) -> dict:
        """
        【V71.0 · 终极生产版】(生产环境清洁版)
        """
        intraday_data = context['intraday_data']
        hf_analysis_df = context['hf_analysis_df']
        metrics = {}
        if not hf_analysis_df.empty and 'main_force_ofi' in hf_analysis_df.columns and 'retail_ofi' in hf_analysis_df.columns:
            mf_ofi_series = hf_analysis_df['main_force_ofi']
            retail_ofi_series = hf_analysis_df['retail_ofi']
            if mf_ofi_series.var() > 0 and retail_ofi_series.var() > 0:
                correlation = mf_ofi_series.corr(retail_ofi_series)
                metrics['mf_retail_liquidity_swap_corr'] = correlation
        else:
            if 'main_force_net_vol' in intraday_data.columns and 'retail_net_vol' in intraday_data.columns:
                mf_net_series = intraday_data['main_force_net_vol']
                retail_net_series = intraday_data['retail_net_vol']
                if mf_net_series.var() != 0 and retail_net_series.var() != 0 and len(mf_net_series) > 1:
                    rolling_corr = mf_net_series.rolling(window=30).corr(retail_net_series)
                    metrics['mf_retail_liquidity_swap_corr'] = rolling_corr.mean()
        return metrics

    @staticmethod
    def _calculate_retail_sentiment_metrics(context: dict) -> dict:
        intraday_data = context['intraday_data']
        hf_analysis_df = context['hf_analysis_df']
        daily_data = context['daily_data']
        common_data = context['common_data']
        should_probe = context['debug']['should_probe']
        stock_code = context['debug']['stock_code']
        current_date = context['daily_data'].name.date()
        from datetime import time
        import pandas as pd
        import numpy as np
        metrics = {
            'retail_fomo_premium_index': np.nan,
            'retail_panic_surrender_index': np.nan
        }
        day_high, day_low = common_data['day_high'], common_data['day_low']
        atr = common_data['atr']
        if not hf_analysis_df.empty and pd.notna(atr) and atr > 0:
            # --- 零售 FOMO 溢价指数 ---
            # 修正 is_new_high 的 fillna 策略
            hf_analysis_df['is_new_high'] = hf_analysis_df['price'] > hf_analysis_df['price'].cummax().shift(1).fillna(method='bfill')
            # 直接筛选在创新高时发生的零售买入交易
            retail_buy_trades_at_new_high = hf_analysis_df[
                (hf_analysis_df['amount'] < 50000) &
                (hf_analysis_df['type'] == 'B') &
                (hf_analysis_df['is_new_high'])
            ].copy()
            if not retail_buy_trades_at_new_high.empty:
                total_weighted_fomo_score = 0
                total_fomo_volume = 0
                # 计算创新高零售买入的平均单笔量，作为 volume_spike_component 的归一化基准
                avg_retail_trade_vol_at_new_high = retail_buy_trades_at_new_high['volume'].mean()
                cost_mf_sell = daily_data.get('avg_cost_main_sell')
                if pd.notna(cost_mf_sell) and cost_mf_sell > 0:
                    for _, trade in retail_buy_trades_at_new_high.iterrows():
                        fomo_vol_in_event = trade['volume']
                        cost_fomo = trade['price']
                        cost_premium_component = (cost_fomo - cost_mf_sell) / atr
                        aggressive_buy = (trade['price'] >= trade['sell_price1'])
                        aggression_component = 1.0 if aggressive_buy else 0.5
                        volume_spike_component = 0.0
                        if pd.notna(avg_retail_trade_vol_at_new_high) and avg_retail_trade_vol_at_new_high > 0:
                            volume_spike_component = np.log1p(trade['volume'] / avg_retail_trade_vol_at_new_high)
                        event_fomo_score = cost_premium_component * aggression_component * volume_spike_component
                        total_weighted_fomo_score += event_fomo_score * fomo_vol_in_event
                        total_fomo_volume += fomo_vol_in_event
                if total_fomo_volume > 0:
                    weighted_avg_fomo_score = total_weighted_fomo_score / total_fomo_volume
                    metrics['retail_fomo_premium_index'] = weighted_avg_fomo_score * 100
            # --- 零售恐慌投降指数 ---
            # 修正 is_new_low 的 fillna 策略
            hf_analysis_df['is_new_low'] = hf_analysis_df['price'] < hf_analysis_df['price'].cummin().shift(1).fillna(method='bfill')
            # 直接筛选在创新低时发生的零售卖出交易
            retail_sell_trades_at_new_low = hf_analysis_df[
                (hf_analysis_df['amount'] < 50000) &
                (hf_analysis_df['type'] == 'S') &
                (hf_analysis_df['is_new_low'])
            ].copy()
            if not retail_sell_trades_at_new_low.empty:
                total_weighted_panic_score = 0
                total_panic_volume = 0
                # 计算创新低零售卖出的平均单笔量，作为 volume_spike_component 的归一化基准
                avg_retail_trade_vol_at_new_low = retail_sell_trades_at_new_low['volume'].mean()
                cost_mf_buy = daily_data.get('avg_cost_main_buy')
                if pd.notna(cost_mf_buy) and cost_mf_buy > 0:
                    for _, trade in retail_sell_trades_at_new_low.iterrows():
                        panic_vol_in_event = trade['volume']
                        cost_panic = trade['price']
                        cost_discount_component = (cost_mf_buy - cost_panic) / atr
                        aggressive_sell = (trade['price'] <= trade['buy_price1'])
                        aggression_component = 1.0 if aggressive_sell else 0.5
                        volume_spike_component = 0.0
                        if pd.notna(avg_retail_trade_vol_at_new_low) and avg_retail_trade_vol_at_new_low > 0:
                            volume_spike_component = np.log1p(trade['volume'] / avg_retail_trade_vol_at_new_low)
                        event_panic_score = cost_discount_component * aggression_component * volume_spike_component
                        total_weighted_panic_score += event_panic_score * panic_vol_in_event
                        total_panic_volume += panic_vol_in_event
                if total_panic_volume > 0:
                    weighted_avg_panic_score = total_weighted_panic_score / total_panic_volume
                    metrics['retail_panic_surrender_index'] = weighted_avg_panic_score * 100
            continuous_trading_df = intraday_data[intraday_data.index.time < time(14, 57)].copy()
            if pd.notna(day_high) and pd.notna(day_low):
                day_range = day_high - day_low
                if day_range > 0:
                    fomo_zone_threshold = day_low + 0.75 * day_range
                    fomo_zone_df = continuous_trading_df[continuous_trading_df['minute_vwap'] > fomo_zone_threshold]
                    if not fomo_zone_df.empty and 'retail_net_vol' in fomo_zone_df.columns and 'retail_buy_vol' in continuous_trading_df.columns and 'minute_vwap' in fomo_zone_df.columns:
                        fomo_retail_df = fomo_zone_df[fomo_zone_df['retail_net_vol'] > 0]
                        if not fomo_retail_df.empty:
                            fomo_vol = fomo_retail_df['retail_net_vol'].sum()
                            total_retail_buy_vol = continuous_trading_df[continuous_trading_df['retail_buy_vol'] > 0]['retail_buy_vol'].sum()
                            if fomo_vol > 0 and total_retail_buy_vol > 0:
                                cost_fomo = (fomo_retail_df['minute_vwap'] * fomo_retail_df['retail_net_vol']).sum() / fomo_vol
                                cost_mf_sell = daily_data.get('avg_cost_main_sell')
                                if pd.notna(cost_mf_sell) and cost_mf_sell > 0:
                                    premium = (cost_fomo / cost_mf_sell - 1)
                                    metrics['retail_fomo_premium_index'] = premium * (fomo_vol / total_retail_buy_vol) * 100
                    panic_zone_threshold = day_low + 0.25 * day_range
                    panic_zone_df = continuous_trading_df[continuous_trading_df['minute_vwap'] < panic_zone_threshold]
                    if not panic_zone_df.empty and 'retail_net_vol' in panic_zone_df.columns and 'retail_sell_vol' in continuous_trading_df.columns and 'minute_vwap' in panic_zone_df.columns:
                        panic_retail_df = panic_zone_df[panic_zone_df['retail_net_vol'] < 0]
                        if not panic_retail_df.empty:
                            panic_vol = abs(panic_retail_df['retail_net_vol'].sum())
                            total_retail_sell_vol = continuous_trading_df[continuous_trading_df['retail_sell_vol'] > 0]['retail_sell_vol'].sum()
                            if panic_vol > 0 and total_retail_sell_vol > 0:
                                cost_panic = (panic_retail_df['minute_vwap'] * abs(panic_retail_df['retail_net_vol'])).sum() / panic_vol
                                cost_mf_buy = daily_data.get('avg_cost_main_buy')
                                if pd.notna(cost_mf_buy) and cost_mf_buy > 0:
                                    discount = (cost_mf_buy - cost_panic) / cost_mf_buy
                                    metrics['retail_panic_surrender_index'] = discount * (panic_vol / total_retail_sell_vol) * 100
        return metrics

    @staticmethod
    def _calculate_panic_cascade_metrics(context: dict) -> dict:
        """
        【V71.0 · 终极生产版】(生产环境清洁版)
        【V72.0 · 资金流拆分版】
        - 核心增强: 拆分 `panic_selling_cascade` 为买卖双方贡献。
        """
        intraday_data = context['intraday_data']
        hf_analysis_df = context['hf_analysis_df']
        common_data = context['common_data']
        from scipy.signal import find_peaks
        from datetime import time
        import numpy as np
        metrics = {}
        atr = common_data['atr']
        continuous_trading_df = intraday_data[intraday_data.index.time < time(14, 57)].copy()
        if not continuous_trading_df.empty and 'minute_vwap' in continuous_trading_df.columns and pd.notna(atr) and atr > 0:
            peaks, _ = find_peaks(continuous_trading_df['minute_vwap'].values)
            troughs, _ = find_peaks(-continuous_trading_df['minute_vwap'].values)
            turning_points = sorted(list(set(np.concatenate(([0], troughs, peaks, [len(continuous_trading_df)-1])))))
            if not hf_analysis_df.empty:
                total_weighted_panic_score = 0
                total_price_drop = 0
                total_retail_sell_vol_sum = 0 # 新增行
                total_mf_buy_vol_sum = 0 # 新增行
                for i in range(len(turning_points) - 1):
                    start_idx, end_idx = turning_points[i], turning_points[i+1]
                    window_df = continuous_trading_df.iloc[start_idx:end_idx+1]
                    if window_df.empty or len(window_df) < 2: continue
                    if window_df['minute_vwap'].iloc[-1] < window_df['minute_vwap'].iloc[0]:
                        start_time, end_time = window_df.index[0], window_df.index[-1]
                        panic_hf_df = hf_analysis_df[(hf_analysis_df.index >= start_time) & (hf_analysis_df.index <= end_time)]
                        if not panic_hf_df.empty:
                            price_drop_in_leg = window_df['minute_vwap'].iloc[0] - window_df['minute_vwap'].iloc[-1]
                            total_price_drop += price_drop_in_leg
                            price_impact_component = price_drop_in_leg / atr
                            ask_depth = panic_hf_df[[f'sell_volume{i}' for i in range(1, 6)]].sum(axis=1).mean()
                            bid_depth = panic_hf_df[[f'buy_volume{i}' for i in range(1, 6)]].sum(axis=1).mean()
                            liquidity_vacuum_component = np.tanh(np.log1p(ask_depth / bid_depth)) if bid_depth > 0 else 1.0
                            retail_trades_in_leg = panic_hf_df[panic_hf_df['amount'] < 50000]
                            retail_sell_trades = retail_trades_in_leg[retail_trades_in_leg['type'] == 'S']
                            total_retail_sell_vol = retail_sell_trades['volume'].sum()
                            total_retail_sell_vol_sum += total_retail_sell_vol # 新增行
                            mf_buy_trades_in_leg = panic_hf_df[(panic_hf_df['amount'] > 200000) & (panic_hf_df['type'] == 'B')] # 新增行
                            total_mf_buy_vol = mf_buy_trades_in_leg['volume'].sum() # 新增行
                            total_mf_buy_vol_sum += total_mf_buy_vol # 新增行
                            if total_retail_sell_vol > 0:
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
                    # 新增拆分指标
                    metrics['panic_sell_volume_contribution'] = (total_retail_sell_vol_sum / common_data['daily_total_volume']) * 100 if common_data['daily_total_volume'] > 0 else np.nan # 新增行
                    metrics['panic_buy_absorption_contribution'] = (total_mf_buy_vol_sum / common_data['daily_total_volume']) * 100 if common_data['daily_total_volume'] > 0 else np.nan # 新增行
            else:
                panic_vol, total_panic_vol = 0, 0
                # Fallback for split metrics if hf_analysis_df is empty
                total_retail_sell_vol_fallback = 0 # 新增行
                total_mf_buy_vol_fallback = 0 # 新增行
                for i in range(len(turning_points) - 1):
                    start_idx, end_idx = turning_points[i], turning_points[i+1]
                    window_df = continuous_trading_df.iloc[start_idx:end_idx+1]
                    if window_df.empty or len(window_df) < 2 or 'minute_vwap' not in window_df.columns or 'main_force_net_vol' not in window_df.columns or 'vol_shares' not in window_df.columns:
                        continue
                    if window_df['minute_vwap'].iloc[-1] <= window_df['minute_vwap'].iloc[0]:
                        total_panic_vol += window_df['vol_shares'].sum()
                        mf_net_vol = window_df['main_force_net_vol'].sum()
                        if mf_net_vol < 0:
                            panic_vol += abs(mf_net_vol)
                        # Fallback for split metrics
                        total_retail_sell_vol_fallback += window_df['retail_sell_vol'].sum() # 新增行
                        total_mf_buy_vol_fallback += window_df['main_force_buy_vol'].sum() # 新增行
                if total_panic_vol > 0:
                    metrics['panic_selling_cascade'] = (panic_vol / total_panic_vol) * 100
                    metrics['panic_sell_volume_contribution'] = (total_retail_sell_vol_fallback / common_data['daily_total_volume']) * 100 if common_data['daily_total_volume'] > 0 else np.nan # 新增行
                    metrics['panic_buy_absorption_contribution'] = (total_mf_buy_vol_fallback / common_data['daily_total_volume']) * 100 if common_data['daily_total_volume'] > 0 else np.nan # 新增行
        return metrics

    @staticmethod
    def _calculate_misc_minute_metrics(context: dict) -> dict:
        """
        【V71.0 · 终极生产版】(生产环境清洁版)
        """
        intraday_data = context['intraday_data']
        hf_analysis_df = context['hf_analysis_df']
        common_data = context['common_data']
        from datetime import time
        import numpy as np
        import pandas as pd
        metrics = {}
        day_open, day_close = common_data['day_open'], common_data['day_close']
        atr = common_data['atr']
        daily_total_volume = common_data['daily_total_volume']
        if not hf_analysis_df.empty and 'main_force_ofi' in hf_analysis_df.columns:
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
            continuous_trading_df = intraday_data[intraday_data.index.time < time(14, 57)].copy()
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
        【V68.2 · 阻力升维终版】(生产环境清洁版)
        - 核心逻辑: 废弃“弹性”概念，转向更本质的“阻力”概念 (abs(总净主动量) / abs(总价差))，
                     确保 `asymmetric_friction_index` 在任何市场场景下都具有数学鲁棒性。
        """
        hf_analysis_df = context['hf_analysis_df']
        import numpy as np
        import pandas as pd
        metrics = {
            'micro_impact_elasticity': np.nan,
            'price_reversion_velocity': np.nan,
            'asymmetric_friction_index': np.nan,
        }
        if hf_analysis_df.empty or 'mid_price_delta' not in hf_analysis_df.columns:
            return metrics
        elasticity_series = hf_analysis_df['mid_price_delta'] / hf_analysis_df['net_active_volume'].replace(0, np.nan)
        weights_vol = hf_analysis_df['volume']
        valid_elasticity = elasticity_series.dropna()
        if not valid_elasticity.empty and weights_vol.sum() > 0:
            metrics['micro_impact_elasticity'] = np.average(valid_elasticity, weights=weights_vol[elasticity_series.notna()])
        hf_analysis_df['next_mid_price_delta'] = hf_analysis_df['mid_price_delta'].shift(-1)
        reversion_df = hf_analysis_df.dropna(subset=['mid_price_delta', 'next_mid_price_delta'])
        if not reversion_df.empty and reversion_df['mid_price_delta'].var() > 0 and reversion_df['next_mid_price_delta'].var() > 0:
            reversion_product = -reversion_df['mid_price_delta'] * reversion_df['next_mid_price_delta']
            reversion_signal = np.sign(reversion_product)
            weights_rev = reversion_df['volume']
            if weights_rev.sum() > 0:
                metrics['price_reversion_velocity'] = np.average(reversion_signal, weights=weights_rev) * 100
        up_moves = hf_analysis_df[hf_analysis_df['mid_price_delta'] > 0]
        down_moves = hf_analysis_df[hf_analysis_df['mid_price_delta'] < 0]
        if not up_moves.empty and not down_moves.empty:
            up_price_delta_sum = up_moves['mid_price_delta'].sum()
            up_net_vol_abs_sum = abs(up_moves['net_active_volume'].sum())
            down_price_delta_abs_sum = abs(down_moves['mid_price_delta'].sum())
            down_net_vol_abs_sum = abs(down_moves['net_active_volume'].sum())
            upward_resistance = up_net_vol_abs_sum / up_price_delta_sum if up_price_delta_sum > 0 else np.nan
            downward_resistance = down_net_vol_abs_sum / down_price_delta_abs_sum if down_price_delta_abs_sum > 0 else np.nan
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





