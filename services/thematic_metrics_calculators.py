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
        """
        【V46.0 · 潜龙在渊】
        - 核心新增: 新增战略级指标 `equilibrium_compression_index` (均衡压缩指数)。
                     该指标专门用于量化“内含日”(Inside Day)的能量压缩程度，通过综合评估
                     空间、位置与力量三个维度，旨在识别重大突破前的“潜龙在渊”状态。
        """
        group = context['group']
        continuous_group = context['continuous_group']
        tick_df = context.get('tick_df')
        day_high_qfq = context['day_high_qfq'] # 新增
        day_low_qfq = context['day_low_qfq'] # 新增
        day_close_qfq = context['day_close_qfq']
        total_volume_safe = context['total_volume_safe'] # 新增
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
                results['vpoc_consensus_strength'] = vp_hf.max() / total_volume
                vp_prob = vp_hf[vp_hf > 0] / total_volume
                entropy = -np.sum(vp_prob * np.log2(vp_prob))
                max_entropy = np.log2(len(vp_prob))
                results['volume_profile_entropy'] = entropy / max_entropy if max_entropy > 0 else 0.0
                if enable_probe and is_target_date:
                    print(f"--- [探针 ASM.{trade_date_str}] market_profile (高频) ---")
                    print(f"    - 原料: {len(tick_df)} 笔Tick数据")
                    print(f"    - 节点: 高精度VPOC={today_vpoc:.2f}, VPOC成交量占比={results['vpoc_consensus_strength']:.2%}")
                    print(f"    -> 结果 (剖面熵): {results.get('volume_profile_entropy', np.nan):.4f}")
        if continuous_group['vol'].sum() > 0:
            vp_minute = continuous_group.groupby(pd.cut(continuous_group['close'], bins=20, duplicates='drop'))['vol'].sum()
            if pd.isna(today_vpoc) and not vp_minute.empty:
                vpoc_interval = vp_minute.idxmax()
                today_vpoc = vpoc_interval.mid if pd.notna(vpoc_interval) else day_close_qfq
                results['vpoc_consensus_strength'] = vp_minute.max() / continuous_group['vol'].sum()
            vpoc_interval_for_va = vp_minute.idxmax() if not vp_minute.empty else np.nan
            today_vah, today_val = ThematicMetricsCalculators._calculate_value_area(vp_minute, continuous_group['vol'].sum(), vpoc_interval_for_va)
        else:
            today_vah, today_val = np.nan, np.nan
        if pd.notna(atr_14) and atr_14 > 0 and pd.notna(today_vpoc):
            deviation_magnitude = (day_close_qfq - today_vpoc) / atr_14
            results['vpoc_deviation_magnitude'] = deviation_magnitude
            tail_period_df = group[group.index.time >= time(14, 45)]
            if not tail_period_df.empty and not continuous_group.empty and continuous_group['vol'].mean() > 0:
                tail_force_factor = np.log1p(tail_period_df['vol'].mean() / continuous_group['vol'].mean())
                results['closing_conviction_score'] = deviation_magnitude * tail_force_factor
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
        # 新增代码块：计算 `equilibrium_compression_index`
        prev_high = prev_day_metrics.get('high')
        prev_low = prev_day_metrics.get('low')
        prev_volume = prev_day_metrics.get('volume')
        if all(pd.notna(v) for v in [prev_high, prev_low, prev_vpoc, prev_volume, today_vpoc]):
            if day_high_qfq <= prev_high and day_low_qfq >= prev_low: # 判断是否为内含日
                prev_range = prev_high - prev_low
                today_range = day_high_qfq - day_low_qfq
                if prev_range > 0 and prev_volume > 0:
                    space_compression = 1 - (today_range / prev_range)
                    positional_balance = 1 - (abs(today_vpoc - prev_vpoc) / prev_range)
                    volume_intensity = np.tanh((total_volume_safe / prev_volume) - 1)
                    score = space_compression * positional_balance * (1 + volume_intensity)
                    results['equilibrium_compression_index'] = score
                    if enable_probe and is_target_date:
                        print(f"--- [探针 ASM.{trade_date_str}] equilibrium_compression_index (潜龙在渊) ---")
                        print(f"    - 前置: 内含日确认 (今日 {day_low_qfq:.2f}-{day_high_qfq:.2f} vs 昨日 {prev_low:.2f}-{prev_high:.2f})")
                        print(f"    - 维度1 (空间压缩): 1 - ({today_range:.2f} / {prev_range:.2f}) = {space_compression:.4f}")
                        print(f"    - 维度2 (位置均衡): 1 - (|{today_vpoc:.2f} - {prev_vpoc:.2f}| / {prev_range:.2f}) = {positional_balance:.4f}")
                        print(f"    - 维度3 (力量胶着): tanh({total_volume_safe:,.0f} / {prev_volume:,.0f} - 1) = {volume_intensity:.4f}")
                        print(f"    - 计算: {space_compression:.4f} * {positional_balance:.4f} * (1 + {volume_intensity:.4f})")
                        print(f"    -> 结果: {score:.4f}")
        results['_today_vpoc'] = today_vpoc
        results['_today_vah'] = today_vah
        results['_today_val'] = today_val
        return results

    @staticmethod
    def calculate_forward_looking_metrics(context: dict) -> dict:
        """
        【V37.4 · 张力奇点修正】
        - 核心修复: 修正了 `auction_showdown_score` 中盘口张力因子的数学模型。
                     使用 `np.exp(pre_auction_tension)` 替代了原有的 `(1 + pre_auction_tension)`，
                     解决了当卖压巨大(张力->-1)时指标值被强制归零的逻辑“奇点”问题，
                     确保指标在极端压力下依然能做出有效评判。
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
                        # 使用 exp 函数替换 (1 + tension) 因子，消除奇点
                        tension_factor = np.exp(pre_auction_tension)
                        results['auction_showdown_score'] = auction_price_change * np.log1p(volume_surprise_factor) * tension_factor
                        if enable_probe and is_target_date:
                            print(f"--- [探针 ASM.{trade_date_str}] auction_showdown_score (高频) ---")
                            print(f"    - 维度1 (价变): 收盘价={day_close_qfq:.2f}, 竞价前价={close_before_auction:.2f} -> {auction_price_change:.2f}%")
                            print(f"    - 维度2 (量能意外): 竞价成交={auction_volume:,.0f}, 前30min均量={avg_vol_pre_auction:,.0f} -> {volume_surprise_factor:.2f}倍")
                            # 更新探针日志以反映新的张力因子模型
                            print(f"    - 维度3 (盘口张力): 竞价前买一量={last_snapshot.get('buy_volume1', 0) if last_snapshot is not None else 0:,.0f}, 卖一量={last_snapshot.get('sell_volume1', 0) if last_snapshot is not None else 0:,.0f} -> {pre_auction_tension:.4f}")
                            print(f"    - 节点 (张力因子): exp({pre_auction_tension:.4f}) = {tension_factor:.4f}")
                            print(f"    - 计算: {auction_price_change:.2f} * log1p({volume_surprise_factor:.2f}) * {tension_factor:.4f}")
                            print(f"    -> 结果: {results['auction_showdown_score']:.4f}")
                    else:
                        avg_vol_minute_continuous = continuous_group['vol'].mean()
                        if avg_vol_minute_continuous > 0:
                            auction_volume_multiple = (auction_period_df['vol'].sum() / 3) / avg_vol_minute_continuous
                            results['auction_showdown_score'] = auction_price_change * np.log1p(auction_volume_multiple)
                            if enable_probe and is_target_date:
                                print(f"--- [探针 ASM.{trade_date_str}] auction_showdown_score (分钟降级) ---")
                                print(f"    -> 结果: {results['auction_showdown_score']:.4f}")
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
        【V51.0 · 赏罚同源】
        - 核心修正: 对 `defense_solidity_score` 的计算公式进行了镜像修正，使其与 `breakthrough_conviction_score`
                     的逻辑完全对称。新的公式 `Rejection * (1 + Purity)` 确保了当防守失败、收盘于
                     支撑之下时，指标能正确输出负值，从而实现了对无效抵抗的精准惩罚。
        - 核心修正: 重塑 `breakthrough_conviction_score` 的计算公式，从 `Purity * (1 + Confirmation)`
                     改为 `Confirmation * (1 + Purity)`。此举确保了当突破失败、收盘于关隘之下时，
                     最终得分能正确地反映为负值，实现了对“假突破”行为的有效惩戒，使指标赏罚分明。
        """
        continuous_group = context['continuous_group']
        tick_df = context.get('tick_df')
        day_open_qfq = context['day_open_qfq']
        day_high_qfq = context['day_high_qfq']
        day_low_qfq = context['day_low_qfq']
        day_close_qfq = context['day_close_qfq']
        atr_14 = context['atr_14']
        prev_day_metrics = context.get('prev_day_metrics', {})
        total_volume_safe = context['total_volume_safe']
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
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
            tail_df = continuous_group[continuous_group.index.time >= time(14, 0)]
            if not tail_df.empty and pd.notna(atr_14) and atr_14 > 0 and total_volume_safe > 0:
                vwap_tail = (tail_df['amount'].sum() / tail_df['vol'].sum()) if tail_df['vol'].sum() > 0 else np.nan
                vwap_full = (continuous_group['amount'].sum() / continuous_group['vol'].sum()) if continuous_group['vol'].sum() > 0 else np.nan
                if all(pd.notna(v) for v in [vwap_tail, vwap_full]):
                    momentum_deviation = (vwap_tail - vwap_full) / atr_14
                    vol_ratio_tail = tail_df['vol'].sum() / total_volume_safe
                    results['closing_momentum_index'] = momentum_deviation * np.log1p(vol_ratio_tail)
            open_rhythm_df = continuous_group[continuous_group.index.time < time(10, 0)]
            mid_rhythm_df = continuous_group[(continuous_group.index.time >= time(10, 0)) & (continuous_group.index.time < time(14, 30))]
            tail_rhythm_df = continuous_group[continuous_group.index.time >= time(14, 30)]
            if not open_rhythm_df.empty and not mid_rhythm_df.empty and not tail_rhythm_df.empty:
                avg_vol_open = open_rhythm_df['vol'].mean()
                avg_vol_mid = mid_rhythm_df['vol'].mean()
                avg_vol_tail = tail_rhythm_df['vol'].mean()
                avg_vol_ends = (avg_vol_open + avg_vol_tail) / 2
                if avg_vol_ends > 0:
                    results['volume_structure_skew'] = avg_vol_mid / avg_vol_ends
        prev_day_high = prev_day_metrics.get('high')
        if tick_df is not None and not tick_df.empty and pd.notna(prev_day_high) and pd.notna(atr_14) and atr_14 > 0:
            if day_high_qfq > prev_day_high:
                breakthrough_zone_ticks = tick_df[tick_df['price'] >= prev_day_high].copy()
                if not breakthrough_zone_ticks.empty:
                    total_breakthrough_vol = breakthrough_zone_ticks['volume'].sum()
                    if total_breakthrough_vol > 0 and 'price_change' in breakthrough_zone_ticks.columns:
                        self_calc_change = breakthrough_zone_ticks['price'].diff().fillna(0)
                        zero_mask = breakthrough_zone_ticks['price_change'] == 0
                        eff_change = np.where(zero_mask, self_calc_change, breakthrough_zone_ticks['price_change'])
                        net_thrust_vol = (breakthrough_zone_ticks['volume'] * np.sign(eff_change)).sum()
                        breakthrough_thrust_purity = net_thrust_vol / total_breakthrough_vol
                        confirmation_raw = (day_close_qfq - prev_day_high) / atr_14
                        confirmation_factor = np.tanh(confirmation_raw)
                        score = confirmation_factor * (1 + breakthrough_thrust_purity)
                        results['breakthrough_conviction_score'] = score
                        if enable_probe and is_target_date:
                            print(f"--- [探针 ASM.{trade_date_str}] breakthrough_conviction_score (关隘验刃) ---")
                            print(f"    - 关隘: 昨日高点={prev_day_high:.2f}")
                            print(f"    - 节点1 (攻坚动能): 对突破区 {len(breakthrough_zone_ticks)} 笔成交进行动能分析 -> 推力纯度={breakthrough_thrust_purity:.4f}")
                            print(f"    - 节点2 (领土确认): tanh(({day_close_qfq:.2f} - {prev_day_high:.2f}) / {atr_14:.4f}) -> 确认因子={confirmation_factor:.4f}")
                            print(f"    - 计算: {confirmation_factor:.4f} * (1 + {breakthrough_thrust_purity:.4f})")
                            print(f"    - 结果: {score:.4f}")
        prev_day_low = prev_day_metrics.get('low')
        if tick_df is not None and not tick_df.empty and pd.notna(prev_day_low) and pd.notna(atr_14) and atr_14 > 0:
            if day_low_qfq < prev_day_low:
                defense_zone_ticks = tick_df[tick_df['price'] <= prev_day_low].copy()
                if not defense_zone_ticks.empty:
                    total_defense_vol = defense_zone_ticks['volume'].sum()
                    if total_defense_vol > 0 and 'price_change' in defense_zone_ticks.columns:
                        self_calc_change = defense_zone_ticks['price'].diff().fillna(0)
                        zero_mask = defense_zone_ticks['price_change'] == 0
                        eff_change = np.where(zero_mask, self_calc_change, defense_zone_ticks['price_change'])
                        net_thrust_vol = (defense_zone_ticks['volume'] * np.sign(eff_change)).sum()
                        defense_thrust_purity = net_thrust_vol / total_defense_vol
                        rejection_raw = (day_close_qfq - prev_day_low) / atr_14
                        rejection_factor = np.tanh(rejection_raw)
                        # 修改代码行：修正公式，实现与突破指标的镜像对称
                        score = rejection_factor * (1 + defense_thrust_purity)
                        results['defense_solidity_score'] = score
                        if enable_probe and is_target_date:
                            print(f"--- [探针 ASM.{trade_date_str}] defense_solidity_score (金城汤池) ---")
                            print(f"    - 防线: 昨日低点={prev_day_low:.2f}")
                            print(f"    - 节点1 (抵抗动能): 对破位区 {len(defense_zone_ticks)} 笔成交进行动能分析 -> 抵抗纯度={defense_thrust_purity:.4f}")
                            print(f"    - 节点2 (战线反推): tanh(({day_close_qfq:.2f} - {prev_day_low:.2f}) / {atr_14:.4f}) -> 反推因子={rejection_factor:.4f}")
                            # 修改代码行：更新探针日志以反映新公式
                            print(f"    - 计算: {rejection_factor:.4f} * (1 + {defense_thrust_purity:.4f})")
                            print(f"    -> 结果: {score:.4f}")
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
