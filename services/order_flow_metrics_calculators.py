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
        这是一个总调度方法，会调用内部的私有方法来计算各个具体指标。
        """
        tick_df = context.get('tick_df')
        group = context.get('group')
        total_volume_safe = context['total_volume_safe']
        results = {}
        if tick_df is None or tick_df.empty or total_volume_safe == 0:
            return results
        # 分别计算各个订单流指标
        results.update(OrderFlowMetricsCalculators._calculate_ofi_and_sweep_metrics(tick_df, total_volume_safe))
        results.update(OrderFlowMetricsCalculators._calculate_vpin_score(tick_df, total_volume_safe))
        results.update(OrderFlowMetricsCalculators._calculate_absorption_and_distribution(group, tick_df))
        return results

    @staticmethod
    def _calculate_ofi_and_sweep_metrics(tick_df: pd.DataFrame, total_volume: float) -> dict:
        """计算订单流失衡(OFI)和扫单强度相关指标。"""
        metrics = {}
        active_buy_vol = tick_df[tick_df['side'] == 'B']['volume'].sum()
        active_sell_vol = tick_df[tick_df['side'] == 'S']['volume'].sum()
        # 累计订单流失衡
        ofi = active_buy_vol - active_sell_vol
        metrics['order_flow_imbalance_score'] = ofi / total_volume
        # 扫单强度
        # 假设扫单定义为单笔成交量大于当日平均单笔成交量的5倍
        avg_tick_vol = tick_df['volume'].mean()
        sweep_threshold = avg_tick_vol * 5
        buy_sweeps = tick_df[(tick_df['side'] == 'B') & (tick_df['volume'] > sweep_threshold)]
        sell_sweeps = tick_df[(tick_df['side'] == 'S') & (tick_df['volume'] > sweep_threshold)]
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
        tick_df['signed_volume'] = tick_df['volume'] * np.where(tick_df['side'] == 'B', 1, -1)
        
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

    @staticmethod
    def _calculate_absorption_and_distribution(group: pd.DataFrame, tick_df: pd.DataFrame) -> dict:
        """计算下跌中的吸收强度和上涨中的派发压力。"""
        metrics = {}
        group['price_change'] = group['close'].diff().fillna(0)
        
        # 找到所有下跌的分钟
        falling_minutes = group[group['price_change'] < 0].index
        # 找到所有上涨的分钟
        rising_minutes = group[group['price_change'] > 0].index
        
        # 筛选出在这些分钟内发生的tick
        ticks_in_falling_minutes = tick_df[tick_df.index.floor('T').isin(falling_minutes)]
        ticks_in_rising_minutes = tick_df[tick_df.index.floor('T').isin(rising_minutes)]
        
        # 计算吸收强度
        if not ticks_in_falling_minutes.empty:
            absorption_buy = ticks_in_falling_minutes[ticks_in_falling_minutes['side'] == 'B']['volume'].sum()
            absorption_sell = ticks_in_falling_minutes[ticks_in_falling_minutes['side'] == 'S']['volume'].sum()
            if absorption_sell > 0:
                metrics['absorption_strength_index'] = absorption_buy / absorption_sell

        # 计算派发压力
        if not ticks_in_rising_minutes.empty:
            distribution_sell = ticks_in_rising_minutes[ticks_in_rising_minutes['side'] == 'S']['volume'].sum()
            distribution_buy = ticks_in_rising_minutes[ticks_in_rising_minutes['side'] == 'B']['volume'].sum()
            if distribution_buy > 0:
                metrics['distribution_pressure_index'] = distribution_sell / distribution_buy
                
        return metrics
