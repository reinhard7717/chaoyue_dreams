from datetime import time
import numpy as np
import pandas as pd

class DerivativeMetricsCalculator:
    """
    【V36.0 · 动能背离】
    衍生指标计算器，专注于对已有的基础指标进行二次分析，以发现更深层次的模式，如“背离”。
    """
    @staticmethod
    def calculate_divergence_metrics(context: dict) -> dict:
        """
        计算各类背离指标，核心是 `price_thrust_divergence`。
        【V37.0 · 背离逻辑修正】
        - 核心修复: 增加 `am_thrust > 0` 的前置条件，确保只在上午为多头主导的情况下才判断背离。
                     这可以过滤掉因空头回补等因素造成的伪背离信号，大幅提升指标的信噪比。
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
        # 1. 计算上下半场的主动买盘强度 (Thrust Purity)
        am_buy_vol = am_ticks[am_ticks['type'] == 'B']['volume'].sum()
        am_sell_vol = am_ticks[am_ticks['type'] == 'S']['volume'].sum()
        am_total_vol = am_buy_vol + am_sell_vol
        am_thrust = (am_buy_vol - am_sell_vol) / am_total_vol if am_total_vol > 0 else 0
        pm_buy_vol = pm_ticks[pm_ticks['type'] == 'B']['volume'].sum()
        pm_sell_vol = pm_ticks[pm_ticks['type'] == 'S']['volume'].sum()
        pm_total_vol = pm_buy_vol + pm_sell_vol
        pm_thrust = (pm_buy_vol - pm_sell_vol) / pm_total_vol if pm_total_vol > 0 else 0
        # 2. 获取上下半场的价格高点
        am_high = am_group['high'].max()
        pm_high = pm_group['high'].max()
        # 3. 计算背离指数
        # 修改代码行：增加 am_thrust > 0 的前置条件，确保背离判断的逻辑严谨性
        if pm_high > am_high and am_thrust > 0 and pm_thrust < am_thrust:
            price_change_pct = (pm_high / am_high - 1)
            thrust_change_pct = (pm_thrust - am_thrust) / abs(am_thrust) if am_thrust != 0 else -1.0
            # 价格涨幅越大，而买盘强度跌幅越大，背离信号越强（负值越深）
            if thrust_change_pct < 0:
                results['price_thrust_divergence'] = price_change_pct / abs(thrust_change_pct) * -1
            else:
                results['price_thrust_divergence'] = 0.0
            if enable_probe and is_target_date:
                print(f"--- [探针 ASM.{trade_date_str}] price_thrust_divergence (背离) ---")
                print(f"    - 上半场: 高点={am_high:.2f}, 买盘强度={am_thrust:.4f}")
                print(f"    - 下半场: 高点={pm_high:.2f}, 买盘强度={pm_thrust:.4f}")
                print(f"    - 节点: 价格变化率={price_change_pct:.4f}, 动能变化率={thrust_change_pct:.4f}")
                print(f"    -> 结果: {results['price_thrust_divergence']:.4f}")
        else:
            results['price_thrust_divergence'] = 0.0 # 无背离则为0
        return results




