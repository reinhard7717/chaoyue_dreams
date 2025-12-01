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
        【V34.0 · 剖面结构穿透】
        - 核心升级: 利用高频Tick数据构建超高分辨率的成交量剖面，从而获得更精确的VPOC、共识强度和剖面熵。
        - 兼容策略: 价值区(VAH/VAL)相关指标暂时沿用分钟数据剖面计算，以保证下游指标链稳定。
        """
        group = context['group']
        continuous_group = context['continuous_group']
        tick_df = context.get('tick_df')
        day_close_qfq = context['day_close_qfq']
        atr_14 = context['atr_14']
        prev_day_metrics = context['prev_day_metrics']
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        results = {}
        today_vpoc = np.nan
        # 高频剖面计算路径
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
        # 分钟降级/兼容路径
        if continuous_group['vol'].sum() > 0:
            vp_minute = continuous_group.groupby(pd.cut(continuous_group['close'], bins=20, duplicates='drop'))['vol'].sum()
            if pd.isna(today_vpoc) and not vp_minute.empty: # 仅在HF路径未计算出VPOC时执行
                vpoc_interval = vp_minute.idxmax()
                today_vpoc = vpoc_interval.mid if pd.notna(vpoc_interval) else day_close_qfq
                results['vpoc_consensus_strength'] = vp_minute.max() / continuous_group['vol'].sum()
            # 价值区(VAH/VAL)指标暂时统一使用分钟数据计算
            vpoc_interval_for_va = vp_minute.idxmax() if not vp_minute.empty else np.nan
            today_vah, today_val = ThematicMetricsCalculators._calculate_value_area(vp_minute, continuous_group['vol'].sum(), vpoc_interval_for_va)
        else: # 如果连分钟数据都没有成交量
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
        【V44.0 · 关隘验刃】
        - 核心新增: 新增战略级指标 `breakthrough_conviction_score` (突破信念分)。
                     该指标通过检验价格突破昨日高点这一关键“关隘”时的动能品质，以及收盘时对
                     新领土的控制程度，来判断一次突破的真实有效性，旨在区分“有效突破”与“假突破”。
        """
        continuous_group = context['continuous_group']
        tick_df = context.get('tick_df') # 新增
        day_open_qfq = context['day_open_qfq']
        day_high_qfq = context['day_high_qfq'] # 新增
        day_close_qfq = context['day_close_qfq']
        atr_14 = context['atr_14']
        prev_day_metrics = context.get('prev_day_metrics', {}) # 新增
        total_volume_safe = context['total_volume_safe']
        debug_info = context.get('debug', {}) # 新增
        is_target_date = debug_info.get('is_target_date', False) # 新增
        enable_probe = debug_info.get('enable_probe', False) # 新增
        trade_date_str = debug_info.get('trade_date_str', 'N/A') # 新增
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
        # 新增代码块：计算 `breakthrough_conviction_score`
        prev_day_high = prev_day_metrics.get('high')
        if tick_df is not None and not tick_df.empty and pd.notna(prev_day_high) and pd.notna(atr_14) and atr_14 > 0:
            if day_high_qfq > prev_day_high:
                breakthrough_zone_ticks = tick_df[tick_df['price'] >= prev_day_high].copy()
                if not breakthrough_zone_ticks.empty:
                    total_breakthrough_vol = breakthrough_zone_ticks['volume'].sum()
                    if total_breakthrough_vol > 0 and 'price_change' in breakthrough_zone_ticks.columns:
                        # 对突破区数据应用“动能回溯”
                        self_calc_change = breakthrough_zone_ticks['price'].diff().fillna(0)
                        zero_mask = breakthrough_zone_ticks['price_change'] == 0
                        eff_change = np.where(zero_mask, self_calc_change, breakthrough_zone_ticks['price_change'])
                        net_thrust_vol = (breakthrough_zone_ticks['volume'] * np.sign(eff_change)).sum()
                        breakthrough_thrust_purity = net_thrust_vol / total_breakthrough_vol
                        # 计算收盘确认因子
                        confirmation_raw = (day_close_qfq - prev_day_high) / atr_14
                        confirmation_factor = np.tanh(confirmation_raw)
                        # 合成最终得分
                        score = breakthrough_thrust_purity * (1 + confirmation_factor)
                        results['breakthrough_conviction_score'] = score
                        if enable_probe and is_target_date:
                            print(f"--- [探针 ASM.{trade_date_str}] breakthrough_conviction_score (关隘验刃) ---")
                            print(f"    - 关隘: 昨日高点={prev_day_high:.2f}")
                            print(f"    - 节点1 (攻坚动能): 对突破区 {len(breakthrough_zone_ticks)} 笔成交进行动能分析 -> 推力纯度={breakthrough_thrust_purity:.4f}")
                            print(f"    - 节点2 (领土确认): tanh(({day_close_qfq:.2f} - {prev_day_high:.2f}) / {atr_14:.4f}) -> 确认因子={confirmation_factor:.4f}")
                            print(f"    - 计算: {breakthrough_thrust_purity:.4f} * (1 + {confirmation_factor:.4f})")
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
