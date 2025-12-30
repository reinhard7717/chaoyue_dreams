# services/thematic_metrics_calculators.py
import pandas as pd
import numpy as np
from datetime import time
from typing import Tuple
from scipy.stats import linregress

class ThematicMetricsCalculators:
    """
    【V28.0 · 主题内核归一】
    - 核心职责: 封装所有基于特定市场理论（如市场剖面、行为金融学）的主题指标计算逻辑。
    - 架构模式: 作为一个无状态的静态工具类，提供一系列独立的、按主题划分的计算函数。
    """
    @staticmethod
    def calculate_market_profile_metrics(context: dict) -> dict:
        """
        【V59.0 · 涤净尘埃】
        - 核心修正: 修正了探针日志中的一个 `KeyError`。移除了对已废弃指标 `vpoc_consensus_strength`
                     的引用，改为在探针内即时计算并显示VPOC点的真实成交量占比，彻底清除了
                     废弃指标的最后残留，确保了代码的健壮性和逻辑一致性。
        - 核心新增: 新增战略级指标 `equilibrium_compression_index` (均衡压缩指数)。
                     该指标专门用于量化“内含日”(Inside Day)的能量压缩程度，通过综合评估
                     空间、位置与力量三个维度，旨在识别重大突破前的“潜龙在渊”状态。
        """
        group = context['group']
        continuous_group = context['continuous_group']
        tick_df = context.get('tick_df')
        day_high_qfq = context['day_high_qfq']
        day_low_qfq = context['day_low_qfq']
        day_close_qfq = context['day_close_qfq']
        total_volume_safe = context['total_volume_safe']
        atr_14 = context['atr_14']
        prev_day_metrics = context['prev_day_metrics']
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        results = {}
        today_vpoc = np.nan
        if tick_df is not None and not tick_df.empty and tick_df['volume'].sum() > 0:
            vp_hf = tick_df.groupby('price')['volume'].sum()
            if not vp_hf.empty:
                today_vpoc = vp_hf.idxmax()
                total_volume = tick_df['volume'].sum()
                vp_prob = vp_hf[vp_hf > 0] / total_volume
                entropy = -np.sum(vp_prob * np.log2(vp_prob))
                max_entropy = np.log2(len(vp_prob))
                results['volume_profile_entropy'] = entropy / max_entropy if max_entropy > 0 else 0.0
        if continuous_group['vol'].sum() > 0:
            # 使用pd.cut进行分箱，处理非唯一边界问题
            try:
                bins = pd.cut(continuous_group['close'], bins=20, duplicates='drop')
                vp_minute = continuous_group.groupby(bins)['vol'].sum()
            except ValueError: # 如果价格范围太小无法分箱
                vp_minute = continuous_group.groupby('close')['vol'].sum()
            if pd.isna(today_vpoc) and not vp_minute.empty:
                vpoc_interval = vp_minute.idxmax()
                today_vpoc = vpoc_interval.mid if hasattr(vpoc_interval, 'mid') else vpoc_interval
            vpoc_interval_for_va = vp_minute.idxmax() if not vp_minute.empty else np.nan
            today_vah, today_val = ThematicMetricsCalculators._calculate_value_area(vp_minute, continuous_group['vol'].sum(), vpoc_interval_for_va)
        else:
            today_vah, today_val = np.nan, np.nan
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
        prev_high = prev_day_metrics.get('high')
        prev_low = prev_day_metrics.get('low')
        prev_volume = prev_day_metrics.get('volume')
        if all(pd.notna(v) for v in [prev_high, prev_low, prev_vpoc, prev_volume, today_vpoc]):
            if day_high_qfq <= prev_high and day_low_qfq >= prev_low:
                prev_range = prev_high - prev_low
                today_range = day_high_qfq - day_low_qfq
                if prev_range > 0 and prev_volume > 0:
                    space_compression = 1 - (today_range / prev_range)
                    positional_balance = 1 - (abs(today_vpoc - prev_vpoc) / prev_range)
                    volume_intensity = np.tanh((total_volume_safe / prev_volume) - 1)
                    score = space_compression * positional_balance * (1 + volume_intensity)
                    results['equilibrium_compression_index'] = score
        results['_today_vpoc'] = today_vpoc
        results['_today_vah'] = today_vah
        results['_today_val'] = today_val
        return results

    @staticmethod
    def calculate_forward_looking_metrics(context: dict) -> dict:
        """
        【V67.0 · 冲击验真】
        - 核心升维: 将 `price_shock_factor` 彻底重构为 `shock_conviction_score` (冲击置信度)。
                     新指标通过融合“基础冲击幅度”、“日内路径效率”与“推力纯度”三大维度，
                     旨在量化价格冲击的“质量”与“可持续性”，从而实现对未来走势更精准的前瞻。
        """
        group = context['group']
        continuous_group = context['continuous_group']
        level5_df = context.get('level5_df')
        day_open_qfq = context['day_open_qfq']
        day_high_qfq = context['day_high_qfq']
        day_low_qfq = context['day_low_qfq']
        day_close_qfq = context['day_close_qfq']
        pre_close_qfq = context['pre_close_qfq']
        atr_5 = context['atr_5']
        atr_14 = context['atr_14']
        atr_50 = context['atr_50']
        intraday_thrust_purity = context.get('intraday_thrust_purity', 0.0) # 获取推力纯度
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        results = {}
        auction_period_df = group[group.index.time >= time(14, 57)]
        if not auction_period_df.empty and not continuous_group.empty:
            close_before_auction_series = continuous_group.loc[continuous_group.index.time < time(14, 57, 0)]['close']
            if not close_before_auction_series.empty:
                close_before_auction = close_before_auction_series.iloc[-1]
                if pd.notna(close_before_auction) and close_before_auction > 0:
                    auction_price_change = (day_close_qfq / close_before_auction - 1) * 100
                    if level5_df is not None and not level5_df.empty:
                        pre_auction_period_df = group[(group.index.time >= time(14, 27)) & (group.index.time < time(14, 57))]
                        avg_vol_pre_auction = pre_auction_period_df['vol'].mean() if not pre_auction_period_df.empty else 0
                        auction_volume = auction_period_df['vol'].sum()
                        volume_surprise_factor = auction_volume / avg_vol_pre_auction if avg_vol_pre_auction > 0 else 1.0
                        last_snapshot_series = level5_df.loc[level5_df.index.time < time(14, 57, 0)]
                        last_snapshot = last_snapshot_series.iloc[-1] if not last_snapshot_series.empty else None
                        pre_auction_tension = 0
                        if last_snapshot is not None:
                            b1_v, a1_v = last_snapshot.get('buy_volume1', 0), last_snapshot.get('sell_volume1', 0)
                            if (b1_v + a1_v) > 0:
                                pre_auction_tension = (b1_v - a1_v) / (b1_v + a1_v)
                        tension_factor = np.exp(pre_auction_tension)
                        results['auction_showdown_score'] = auction_price_change * np.log1p(volume_surprise_factor) * tension_factor
                    else:
                        avg_vol_minute_continuous = continuous_group['vol'].mean()
                        if avg_vol_minute_continuous > 0:
                            auction_volume_multiple = (auction_period_df['vol'].sum() / 3) / avg_vol_minute_continuous
                            results['auction_showdown_score'] = auction_price_change * np.log1p(auction_volume_multiple)
        # 修改代码块：升维为 shock_conviction_score
        if all(pd.notna(v) for v in [day_high_qfq, day_low_qfq, day_open_qfq, day_close_qfq, pre_close_qfq, atr_14]) and atr_14 > 0:
            # 1. 基础冲击：量化冲击的原始幅度
            base_shock = (day_close_qfq - pre_close_qfq) / atr_14
            # 2. 信念因子：融合路径效率与推力纯度
            intraday_range = day_high_qfq - day_low_qfq
            if intraday_range > 0:
                path_efficiency_factor = (day_close_qfq - day_open_qfq) / intraday_range
            else:
                path_efficiency_factor = np.sign(day_close_qfq - day_open_qfq) if day_close_qfq != day_open_qfq else 0
            # 推力纯度因子直接使用
            thrust_purity_factor = intraday_thrust_purity if pd.notna(intraday_thrust_purity) else 0
            # 融合两大信念因子
            conviction_weight = (path_efficiency_factor + thrust_purity_factor)
            # 3. 最终得分：基础冲击 * 信念权重
            results['shock_conviction_score'] = base_shock * conviction_weight
        if all(pd.notna(v) for v in [atr_5, atr_50]) and atr_50 > 0:
            results['volatility_expansion_ratio'] = atr_5 / atr_50
        return results

    @staticmethod
    def calculate_battlefield_metrics(context: dict) -> dict:
        """
        【V69.0 · 决战验刃】
        - 核心升维: `trend_quality_score` 升维为 `trend_acceleration_score`，通过比较上下半场
                     趋势斜率，从“静态质量”的评估转向“动态加速”的捕捉，更擅于发现战局转折点。
        - 核心升维: `closing_momentum_index` 升维为 `final_charge_intensity`，通过对比决战时刻
                     与战前对峙的“推力纯度”，从“状态”的衡量深化为“变化”的洞察，精准量化终场
                     冲锋的决心与烈度。
        """
        continuous_group = context['continuous_group']
        tick_df = context.get('tick_df')
        day_close_qfq = context['day_close_qfq']
        atr_14 = context['atr_14']
        prev_day_metrics = context.get('prev_day_metrics', {})
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        results = {}
        # 1. 升维：趋势加速分 (Trend Acceleration Score)
        if not continuous_group.empty and len(continuous_group) > 1 and pd.notna(atr_14) and atr_14 > 0:
            am_session = continuous_group.between_time('09:30', '11:30')
            pm_session = continuous_group.between_time('13:00', '15:00')
            if len(am_session) > 2 and len(pm_session) > 2:
                y_am = am_session['close'].values
                x_am = np.arange(len(y_am))
                slope_am, _, _, _, _ = linregress(x_am, y_am)
                y_pm = pm_session['close'].values
                x_pm = np.arange(len(y_pm))
                slope_pm, _, _, _, _ = linregress(x_pm, y_pm)
                # 将斜率差用ATR进行标准化
                results['trend_acceleration_score'] = (slope_pm - slope_am) / atr_14
        # 2. 升维：终场冲锋强度 (Final Charge Intensity)
        def _calculate_thrust_purity_for_period(period_df: pd.DataFrame, period_ticks: pd.DataFrame | None) -> float:
            """辅助函数：计算指定时间段的推力纯度，优先高频"""
            if period_ticks is not None and not period_ticks.empty:
                total_vol = period_ticks['volume'].sum()
                if total_vol > 0:
                    # 自行计算价格变动
                    period_ticks['price_change'] = period_ticks['price'].diff().fillna(0)
                    net_thrust_vol = (period_ticks['volume'] * np.sign(period_ticks['price_change'])).sum()
                    return net_thrust_vol / total_vol
            # 降级逻辑
            if not period_df.empty:
                total_vol = period_df['vol'].sum()
                if total_vol > 0:
                    thrust_vector = (period_df['close'] - period_df['open']) * period_df['vol']
                    absolute_energy = abs(period_df['close'] - period_df['open']) * period_df['vol']
                    total_energy = absolute_energy.sum()
                    if total_energy > 0:
                        return thrust_vector.sum() / total_energy
            return 0.0
        pre_charge_df = continuous_group.between_time('13:30', '14:29')
        final_charge_df = continuous_group.between_time('14:30', '15:00')
        pre_charge_ticks = tick_df.between_time('13:30', '14:29') if tick_df is not None else None
        final_charge_ticks = tick_df.between_time('14:30', '15:00') if tick_df is not None else None
        if not pre_charge_df.empty and not final_charge_df.empty:
            purity_pre = _calculate_thrust_purity_for_period(pre_charge_df, pre_charge_ticks)
            purity_final = _calculate_thrust_purity_for_period(final_charge_df, final_charge_ticks)
            vol_pre = pre_charge_df['vol'].sum()
            vol_final = final_charge_df['vol'].sum()
            if vol_pre > 0:
                vol_ratio = vol_final / vol_pre
                results['final_charge_intensity'] = (purity_final - purity_pre) * np.log1p(vol_ratio)
        # 3. 保留：成交结构偏度
        open_rhythm_df = continuous_group.between_time('09:30', '10:00')
        mid_rhythm_df = continuous_group.between_time('10:00', '14:30')
        tail_rhythm_df = continuous_group.between_time('14:30', '15:00')
        if not open_rhythm_df.empty and not mid_rhythm_df.empty and not tail_rhythm_df.empty:
            avg_vol_open = open_rhythm_df['vol'].mean()
            avg_vol_mid = mid_rhythm_df['vol'].mean()
            avg_vol_tail = tail_rhythm_df['vol'].mean()
            # MODIFIED BLOCK START
            # 修正 volume_structure_skew 的计算逻辑，使其能够反映负向偏度
            # 如果盘中成交量相对于两端平均成交量较低，则为负偏度，表示卖压或缺乏买盘
            avg_vol_ends = (avg_vol_open + avg_vol_tail) / 2
            if avg_vol_ends > 1e-9: # 避免除以零
                # 使用 (mid - ends) / ends 的形式，使其可以为负值
                results['volume_structure_skew'] = (avg_vol_mid - avg_vol_ends) / avg_vol_ends
            else:
                results['volume_structure_skew'] = 0.0 # 避免除以零
            # MODIFIED BLOCK END
        # 4. 保留：突破/防守信念分
        prev_day_high = prev_day_metrics.get('high')
        day_high_qfq = context['day_high_qfq']
        if tick_df is not None and not tick_df.empty and pd.notna(prev_day_high) and pd.notna(atr_14) and atr_14 > 0:
            if day_high_qfq > prev_day_high:
                breakthrough_zone_ticks = tick_df[tick_df['price'] >= prev_day_high].copy()
                if not breakthrough_zone_ticks.empty:
                    total_breakthrough_vol = breakthrough_zone_ticks['volume'].sum()
                    if total_breakthrough_vol > 0:
                        breakthrough_zone_ticks['price_change'] = breakthrough_zone_ticks['price'].diff().fillna(0)
                        net_thrust_vol = (breakthrough_zone_ticks['volume'] * np.sign(breakthrough_zone_ticks['price_change'])).sum()
                        breakthrough_thrust_purity = net_thrust_vol / total_breakthrough_vol
                        confirmation_raw = (day_close_qfq - prev_day_high) / atr_14
                        confirmation_factor = np.tanh(confirmation_raw)
                        score = confirmation_factor * (1 + breakthrough_thrust_purity)
                        results['breakthrough_conviction_score'] = score
        prev_day_low = prev_day_metrics.get('low')
        day_low_qfq = context['day_low_qfq']
        if tick_df is not None and not tick_df.empty and pd.notna(prev_day_low) and pd.notna(atr_14) and atr_14 > 0:
            if day_low_qfq < prev_day_low:
                defense_zone_ticks = tick_df[tick_df['price'] <= prev_day_low].copy()
                if not defense_zone_ticks.empty:
                    total_defense_vol = defense_zone_ticks['volume'].sum()
                    if total_defense_vol > 0:
                        defense_zone_ticks['price_change'] = defense_zone_ticks['price'].diff().fillna(0)
                        net_thrust_vol = (defense_zone_ticks['volume'] * np.sign(defense_zone_ticks['price_change'])).sum()
                        defense_thrust_purity = net_thrust_vol / total_defense_vol
                        rejection_raw = (day_close_qfq - prev_day_low) / atr_14
                        rejection_factor = np.tanh(rejection_raw)
                        score = rejection_factor * (1 + defense_thrust_purity)
                        results['defense_solidity_score'] = score
        # 5. 保留：均衡压缩指数
        prev_high = prev_day_metrics.get('high')
        prev_low = prev_day_metrics.get('low')
        prev_volume = prev_day_metrics.get('volume')
        prev_vpoc = prev_day_metrics.get('vpoc')
        today_vpoc = context.get('_today_vpoc')
        total_volume_safe = context['total_volume_safe']
        if all(pd.notna(v) for v in [prev_high, prev_low, prev_vpoc, prev_volume, today_vpoc]):
            if day_high_qfq <= prev_high and day_low_qfq >= prev_low:
                prev_range = prev_high - prev_low
                today_range = day_high_qfq - day_low_qfq
                if prev_range > 0 and prev_volume > 0:
                    space_compression = 1 - (today_range / prev_range)
                    positional_balance = 1 - (abs(today_vpoc - prev_vpoc) / prev_range)
                    volume_intensity = np.tanh((total_volume_safe / prev_volume) - 1)
                    score = space_compression * positional_balance * (1 + volume_intensity)
                    results['equilibrium_compression_index'] = score
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
