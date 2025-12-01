import numpy as np
import pandas as pd
from datetime import time

class OrderFlowMetricsCalculators:
    """
    【V36.0 · 动能背离 · 重构版】
    订单流指标计算器。
    - 核心职责: 封装所有基于订单流微观结构的指标计算，如OFI, VPIN, 扫单强度等。
    - 来源: 本类中的方法是从 `StructuralMetricsCalculators` 中重构剥离而来，以提升代码清晰度和维护性。
    """
    @staticmethod
    def calculate_order_flow_metrics(context: dict) -> dict:
        """
        计算所有与订单流相关的指标。
        【V37.9 · 博弈内核归一】
        - 核心重构: 移除了 `absorption_strength_index` 和 `distribution_pressure_index` 的冗余计算。
                     这两个指标的计算权责已统一收归于 `StructuralMetricsCalculators`。
        """
        tick_df = context.get('tick_df')
        group = context.get('group')
        total_volume_safe = context['total_volume_safe']
        # 移除冗余指标的初始化
        results = {
            'order_flow_imbalance_score': np.nan,
            'buy_sweep_intensity': np.nan,
            'sell_sweep_intensity': np.nan,
            'vpin_score': np.nan,
        }
        if tick_df is None or tick_df.empty or total_volume_safe == 0:
            return results
        results.update(OrderFlowMetricsCalculators._calculate_ofi_and_sweep_metrics(tick_df, total_volume_safe))
        results.update(OrderFlowMetricsCalculators._calculate_vpin_score(tick_df, total_volume_safe))
        # 删除对冗余计算方法的调用
        return results

    @staticmethod
    def _calculate_ofi_and_sweep_metrics(tick_df: pd.DataFrame, total_volume: float) -> dict:
        """
        计算订单流失衡(OFI)和扫单强度相关指标。
        【V42.0 · 动能归一】
        - 核心升级: `order_flow_imbalance_score` 的计算核心从 B/S 盘净量，升级为基于“动能回溯”
                     的真实净推力成交量，使其物理意义与 `intraday_thrust_purity` 完全统一。
        """
        metrics = {}
        # 使用“动能回溯”逻辑计算OFI
        # 累计订单流失衡 (OFI)
        if 'price_change' in tick_df.columns and not tick_df['price_change'].isnull().all():
            self_calculated_change = tick_df['price'].diff().fillna(0)
            zero_change_mask = tick_df['price_change'] == 0
            effective_price_change = np.where(zero_change_mask, self_calculated_change, tick_df['price_change'])
            ofi = (tick_df['volume'] * np.sign(effective_price_change)).sum()
        else: # 回退逻辑
            active_buy_vol = tick_df[tick_df['type'] == 'B']['volume'].sum()
            active_sell_vol = tick_df[tick_df['type'] == 'S']['volume'].sum()
            ofi = active_buy_vol - active_sell_vol
        metrics['order_flow_imbalance_score'] = ofi / total_volume
        # 扫单强度 (逻辑不变，其定义依赖于B/S分类)
        active_buy_vol = tick_df[tick_df['type'] == 'B']['volume'].sum()
        active_sell_vol = tick_df[tick_df['type'] == 'S']['volume'].sum()
        avg_tick_vol = tick_df['volume'].mean()
        sweep_threshold = avg_tick_vol * 5
        buy_sweeps = tick_df[(tick_df['type'] == 'B') & (tick_df['volume'] > sweep_threshold)]
        sell_sweeps = tick_df[(tick_df['type'] == 'S') & (tick_df['volume'] > sweep_threshold)]
        buy_sweep_vol = buy_sweeps['volume'].sum()
        sell_sweep_vol = sell_sweeps['volume'].sum()
        if active_buy_vol > 0:
            metrics['buy_sweep_intensity'] = buy_sweep_vol / active_buy_vol
        if active_sell_vol > 0:
            metrics['sell_sweep_intensity'] = sell_sweep_vol / active_sell_vol
        return metrics

    @staticmethod
    def _calculate_vpin_score(tick_df: pd.DataFrame, total_volume: float) -> dict:
        """计算VPIN（Volume-Synchronized Probability of Informed Trading）指标。"""
        num_buckets = 50
        bucket_size = total_volume / num_buckets
        # 将 'side' 修正为 'type'
        tick_df['signed_volume'] = tick_df['volume'] * np.where(tick_df['type'] == 'B', 1, -1)
        imbalances = []
        current_bucket_vol = 0
        bucket_ofi = 0
        for _, row in tick_df.iterrows():
            current_bucket_vol += row['volume']
            bucket_ofi += row['signed_volume']
            if current_bucket_vol >= bucket_size:
                imbalances.append(bucket_ofi)
                current_bucket_vol = 0
                bucket_ofi = 0
        if imbalances:
            imbalances_abs = np.abs(imbalances)
            vpin = np.sum(imbalances_abs) / (num_buckets * bucket_size)
            return {'vpin_score': vpin}
        return {}







