# services/thematic_metrics_calculators.py
import pandas as pd
import numpy as np
from datetime import time
from scipy.stats import linregress

class ThematicMetricsCalculators:
    """
    【V28.0 · 主题内核归一】
    - 核心职责: 封装所有基于特定市场理论（如市场剖面、行为金融学）的主题指标计算逻辑。
    - 架构模式: 作为一个无状态的静态工具类，提供一系列独立的、按主题划分的计算函数。
    """
    @staticmethod
    def calculate_market_profile_metrics(context: dict) -> dict:
        """计算市场剖面相关指标 (VPOC, 价值区等)"""
        group = context['group']
        continuous_group = context['continuous_group']
        day_close_qfq = context['day_close_qfq']
        atr_14 = context['atr_14']
        prev_day_metrics = context['prev_day_metrics']
        results = {}
        vp = continuous_group.groupby(pd.cut(continuous_group['close'], bins=20, duplicates='drop'))['vol'].sum()
        vpoc_interval = vp.idxmax() if not vp.empty else np.nan
        today_vpoc = vpoc_interval.mid if pd.notna(vpoc_interval) else day_close_qfq
        vpoc_volume_ratio = vp.max() / continuous_group['vol'].sum() if not vp.empty and continuous_group['vol'].sum() > 0 else 0
        if pd.notna(atr_14) and atr_14 > 0:
            deviation_magnitude = (day_close_qfq - today_vpoc) / atr_14
            results['vpoc_deviation_magnitude'] = deviation_magnitude
            results['vpoc_consensus_strength'] = vpoc_volume_ratio
            # 修改代码行：将 group['trade_time'].dt.time 替换为 group.index.time
            tail_period_df = group[group.index.time >= time(14, 45)]
            if not tail_period_df.empty and not continuous_group.empty and continuous_group['vol'].mean() > 0:
                tail_force_factor = np.log1p(tail_period_df['vol'].mean() / continuous_group['vol'].mean())
                results['closing_conviction_score'] = deviation_magnitude * tail_force_factor
        today_vah, today_val = ThematicMetricsCalculators._calculate_value_area(vp, continuous_group['vol'].sum(), vpoc_interval)
        prev_vpoc, prev_atr = prev_day_metrics.get('vpoc'), prev_day_metrics.get('atr_14d')
        if all(pd.notna(v) for v in [today_vpoc, prev_vpoc, prev_atr]) and prev_atr > 0:
            results['value_area_migration'] = (today_vpoc - prev_vpoc) / prev_atr
        prev_vah, prev_val = prev_day_metrics.get('vah'), prev_day_metrics.get('val')
        if all(pd.notna(v) for v in [today_vah, today_val, prev_vah, prev_val]) and (today_vah - today_val) > 0:
            overlap_width = max(0, min(today_vah, prev_vah) - max(today_val, prev_val))
            results['value_area_overlap_pct'] = (overlap_width / (today_vah - today_val)) * 100
        if all(pd.notna(v) for v in [day_close_qfq, today_vpoc, today_vah, today_val]):
            if day_close_qfq > today_vah: results['closing_acceptance_type'] = 2
            elif day_close_qfq > today_vpoc: results['closing_acceptance_type'] = 1
            elif day_close_qfq < today_val: results['closing_acceptance_type'] = -2
            elif day_close_qfq < today_vpoc: results['closing_acceptance_type'] = -1
            else: results['closing_acceptance_type'] = 0
        results['_today_vpoc'] = today_vpoc
        results['_today_vah'] = today_vah
        results['_today_val'] = today_val
        return results

    @staticmethod
    def calculate_forward_looking_metrics(context: dict) -> dict:
        """计算前瞻性与收盘博弈指标"""
        group = context['group']
        continuous_group = context['continuous_group']
        day_open_qfq = context['day_open_qfq']
        day_high_qfq = context['day_high_qfq']
        day_low_qfq = context['day_low_qfq']
        day_close_qfq = context['day_close_qfq']
        pre_close_qfq = context['pre_close_qfq']
        atr_5 = context['atr_5']
        atr_14 = context['atr_14']
        atr_50 = context['atr_50']
        results = {}
        # 修改代码行：将 group['trade_time'].dt.time 替换为 group.index.time
        auction_period_df = group[group.index.time >= time(14, 57)]
        if not auction_period_df.empty and not continuous_group.empty:
            close_before_auction = continuous_group['close'].iloc[-1]
            if pd.notna(close_before_auction) and close_before_auction > 0:
                auction_price_change = (day_close_qfq / close_before_auction - 1) * 100
                avg_vol_minute_continuous = continuous_group['vol'].mean()
                if avg_vol_minute_continuous > 0:
                    auction_volume_multiple = (auction_period_df['vol'].sum() / 3) / avg_vol_minute_continuous
                    results['auction_showdown_score'] = auction_price_change * np.log1p(auction_volume_multiple)
        if all(pd.notna(v) for v in [day_high_qfq, day_low_qfq, pre_close_qfq, atr_14]) and atr_14 > 0:
            true_range = max(day_high_qfq, pre_close_qfq) - min(day_low_qfq, pre_close_qfq)
            shock = true_range / atr_14
            direction = np.sign(day_close_qfq - day_open_qfq) if day_close_qfq != day_open_qfq else 1
            results['price_shock_factor'] = shock * direction
        if all(pd.notna(v) for v in [atr_5, atr_50]) and atr_50 > 0:
            results['volatility_expansion_ratio'] = atr_5 / atr_50
        return results

    @staticmethod
    def calculate_battlefield_metrics(context: dict) -> dict:
        """
        【V31.0 · 索引访问模式统一】
        - 核心修复: 将所有时间过滤从按列访问 `df['trade_time'].dt.time` 统一为按索引访问 `df.index.time`。
        """
        continuous_group = context['continuous_group']
        day_open_qfq = context['day_open_qfq']
        day_close_qfq = context['day_close_qfq']
        atr_14 = context['atr_14']
        total_volume_safe = context['total_volume_safe']
        results = {}
        if not continuous_group.empty and len(continuous_group) > 1:
            vwap_series = continuous_group['minute_vwap'].dropna()
            if len(vwap_series) > 2:
                x = np.arange(len(vwap_series))
                slope, _, r_value, _, _ = linregress(x, vwap_series)
                linearity = r_value**2
                vwap_range = vwap_series.max() - vwap_series.min()
                if vwap_range > 0:
                    if day_close_qfq > day_open_qfq:
                        pullback_control = ((vwap_series - vwap_series.min()) / vwap_range).mean()
                    else:
                        pullback_control = ((vwap_series.max() - vwap_series) / vwap_range).mean()
                    trend_quality = linearity * pullback_control
                    direction = np.sign(day_close_qfq - day_open_qfq) if day_close_qfq != day_open_qfq else 1
                    results['trend_quality_score'] = trend_quality * direction
            # 修改代码行：统一为索引访问
            tail_df = continuous_group[continuous_group.index.time >= time(14, 0)]
            if not tail_df.empty and pd.notna(atr_14) and atr_14 > 0 and total_volume_safe > 0:
                vwap_tail = (tail_df['amount'].sum() / tail_df['vol'].sum()) if tail_df['vol'].sum() > 0 else np.nan
                vwap_full = (continuous_group['amount'].sum() / continuous_group['vol'].sum()) if continuous_group['vol'].sum() > 0 else np.nan
                if all(pd.notna(v) for v in [vwap_tail, vwap_full]):
                    momentum_deviation = (vwap_tail - vwap_full) / atr_14
                    vol_ratio_tail = tail_df['vol'].sum() / total_volume_safe
                    results['closing_momentum_index'] = momentum_deviation * np.log1p(vol_ratio_tail)
            # 修改代码行：统一为索引访问
            open_rhythm_df = continuous_group[continuous_group.index.time < time(10, 0)]
            # 修改代码行：统一为索引访问
            mid_rhythm_df = continuous_group[(continuous_group.index.time >= time(10, 0)) & (continuous_group.index.time < time(14, 30))]
            # 修改代码行：统一为索引访问
            tail_rhythm_df = continuous_group[continuous_group.index.time >= time(14, 30)]
            if not open_rhythm_df.empty and not mid_rhythm_df.empty and not tail_rhythm_df.empty:
                avg_vol_open = open_rhythm_df['vol'].mean()
                avg_vol_mid = mid_rhythm_df['vol'].mean()
                avg_vol_tail = tail_rhythm_df['vol'].mean()
                avg_vol_ends = (avg_vol_open + avg_vol_tail) / 2
                if avg_vol_ends > 0:
                    results['volume_structure_skew'] = avg_vol_mid / avg_vol_ends
        return results

    @staticmethod
    def _calculate_value_area(vp: pd.Series, total_volume: float, vpoc_interval: pd.Interval) -> tuple:
        """计算日内价值区域 (VAH/VAL)"""
        if vp.empty or total_volume == 0 or pd.isna(vpoc_interval):
            return np.nan, np.nan
        value_area_target_volume = total_volume * 0.7
        vp_sorted_by_price = vp.sort_index()
        try:
            poc_idx = vp_sorted_by_price.index.get_loc(vpoc_interval)
        except KeyError:
            return np.nan, np.nan
        current_volume = vp_sorted_by_price.iloc[poc_idx]
        low_idx, high_idx = poc_idx, poc_idx
        while current_volume < value_area_target_volume and (low_idx > 0 or high_idx < len(vp_sorted_by_price) - 1):
            vol_above = vp_sorted_by_price.iloc[high_idx + 1] if high_idx < len(vp_sorted_by_price) - 1 else -1
            vol_below = vp_sorted_by_price.iloc[low_idx - 1] if low_idx > 0 else -1
            if vol_above > vol_below:
                high_idx += 1
                current_volume += vol_above
            else:
                low_idx -= 1
                current_volume += vol_below
        val = vp_sorted_by_price.index[low_idx].left
        vah = vp_sorted_by_price.index[high_idx].right
        return vah, val
