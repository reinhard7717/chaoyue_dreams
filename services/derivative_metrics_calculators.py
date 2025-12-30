from datetime import time
import numpy as np
import pandas as pd
import numba # 确保已导入
from typing import Tuple

@numba.njit(cache=True)
def _numba_calculate_price_thrust_divergence(
    am_high: float, pm_high: float, am_low: float, pm_low: float,
    am_thrust: float, pm_thrust: float
) -> float:
    """
    【Numba优化版】计算价格推力背离。
    """
    top_divergence_score = 0.0
    bottom_divergence_score = 0.0

    # 顶背离计算 (价格新高, 动能减弱)
    if pm_high > am_high and am_thrust > 0 and pm_thrust < am_thrust:
        price_change_pct = (pm_high / am_high - 1)
        thrust_change_pct = (pm_thrust - am_thrust) / abs(am_thrust) if am_thrust != 0 else -1.0
        if thrust_change_pct < 0:
            top_divergence_score = price_change_pct / abs(thrust_change_pct) * -1

    # 底背离计算 (价格新低, 动能增强)
    if pm_low < am_low and am_thrust < 0 and pm_thrust > am_thrust:
        price_change_pct = (am_low / pm_low - 1)
        thrust_change_pct = (pm_thrust - am_thrust) / abs(am_thrust) if am_thrust != 0 else 1.0
        if thrust_change_pct > 0:
            bottom_divergence_score = price_change_pct / abs(thrust_change_pct)
            
    return top_divergence_score + bottom_divergence_score

class DerivativeMetricsCalculator:
    """
    【V36.0 · 动能背离】
    衍生指标计算器，专注于对已有的基础指标进行二次分析，以发现更深层次的模式，如“背离”。
    """
    @staticmethod
    def calculate_divergence_metrics(context: dict) -> dict:
        """
        计算各类背离指标，核心是 `price_thrust_divergence`。
        【V58.0 · 诡道归元】
        - 核心升级: 新增对“底背离”的计算逻辑，实现了对“顶背离”与“底背离”的阴阳合一，
                     使指标能同时捕捉上涨衰竭的风险和下跌企稳的机会。
        - 指标整合: 将顶、底背离得分统一至 `price_thrust_divergence`，负值为顶背离，正值为底背离。
        - 核心优化: 使用Numba优化后的价格推力背离计算函数。
        """
        tick_df = context.get('tick_df')
        group = context.get('group')
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        results = {}
        if tick_df is None or tick_df.empty or group is None or group.empty:
            return results
        midday_break_time = time(13, 0)
        am_ticks = tick_df[tick_df.index.time < midday_break_time]
        pm_ticks = tick_df[tick_df.index.time >= midday_break_time]
        am_group = group[group.index.time < midday_break_time]
        pm_group = group[group.index.time >= midday_break_time]
        if am_ticks.empty or pm_ticks.empty or am_group.empty or pm_group.empty:
            return results
        am_thrust, pm_thrust = 0.0, 0.0 # 确保为浮点数
        am_total_vol = am_ticks['volume'].sum()
        if am_total_vol > 0:
            if 'price_change' in am_ticks.columns and not am_ticks['price_change'].isnull().all():
                self_calculated_change = am_ticks['price'].diff().fillna(0)
                zero_change_mask = am_ticks['price_change'] == 0
                effective_price_change = np.where(zero_change_mask, self_calculated_change, am_ticks['price_change'])
                net_thrust_volume = (am_ticks['volume'] * np.sign(effective_price_change)).sum()
                am_thrust = net_thrust_volume / am_total_vol
            elif 'type' in am_ticks.columns:
                am_buy_vol = am_ticks[am_ticks['type'] == 'B']['volume'].sum()
                am_sell_vol = am_ticks[am_ticks['type'] == 'S']['volume'].sum()
                am_thrust = (am_buy_vol - am_sell_vol) / am_total_vol
        pm_total_vol = pm_ticks['volume'].sum()
        if pm_total_vol > 0:
            if 'price_change' in pm_ticks.columns and not pm_ticks['price_change'].isnull().all():
                self_calculated_change = pm_ticks['price'].diff().fillna(0)
                zero_change_mask = pm_ticks['price_change'] == 0
                effective_price_change = np.where(zero_change_mask, self_calculated_change, pm_ticks['price_change'])
                net_thrust_volume = (pm_ticks['volume'] * np.sign(effective_price_change)).sum()
                pm_thrust = net_thrust_volume / pm_total_vol
            elif 'type' in pm_ticks.columns:
                pm_buy_vol = pm_ticks[pm_ticks['type'] == 'B']['volume'].sum()
                pm_sell_vol = pm_ticks[pm_ticks['type'] == 'S']['volume'].sum()
                pm_thrust = (pm_buy_vol - pm_sell_vol) / pm_total_vol
        am_high, pm_high = am_group['high'].max(), pm_group['high'].max()
        am_low, pm_low = am_group['low'].min(), pm_group['low'].min()
        # 调用Numba优化函数
        results['price_thrust_divergence'] = _numba_calculate_price_thrust_divergence(
            am_high, pm_high, am_low, pm_low, am_thrust, pm_thrust
        )
        return results



