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
        # 新增代码块：高频剖面计算路径
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
        【V35.0 · 收盘博弈穿透】
        - 核心升级: 重构 `auction_showdown_score` 指标，引入“量能意外”和“盘口张力”两大高频变量，
                     深度刻画收盘竞价的博弈心理与主力意图。
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
            close_before_auction = continuous_group.loc[continuous_group.index < time(14, 57, 0)]['close'].iloc[-1]
            if pd.notna(close_before_auction) and close_before_auction > 0:
                auction_price_change = (day_close_qfq / close_before_auction - 1) * 100
                # 新增代码块：高频路径
                if level5_df is not None and not level5_df.empty:
                    pre_auction_period_df = group[(group.index.time >= time(14, 27)) & (group.index.time < time(14, 57))]
                    avg_vol_pre_auction = pre_auction_period_df['vol'].mean() if not pre_auction_period_df.empty else 0
                    auction_volume = auction_period_df['vol'].sum()
                    volume_surprise_factor = auction_volume / avg_vol_pre_auction if avg_vol_pre_auction > 0 else 1.0
                    last_snapshot = level5_df.loc[level5_df.index < time(14, 57, 0)].iloc[-1] if not level5_df.loc[level5_df.index < time(14, 57, 0)].empty else None
                    pre_auction_tension = 0
                    if last_snapshot is not None:
                        b1_v, a1_v = last_snapshot.get('buy_volume1', 0), last_snapshot.get('sell_volume1', 0)
                        if (b1_v + a1_v) > 0:
                            pre_auction_tension = (b1_v - a1_v) / (b1_v + a1_v)
                    results['auction_showdown_score'] = auction_price_change * np.log1p(volume_surprise_factor) * (1 + pre_auction_tension)
                    if enable_probe and is_target_date:
                        print(f"--- [探针 ASM.{trade_date_str}] auction_showdown_score (高频) ---")
                        print(f"    - 维度1 (价变): 收盘价={day_close_qfq:.2f}, 竞价前价={close_before_auction:.2f} -> {auction_price_change:.2f}%")
                        print(f"    - 维度2 (量能意外): 竞价成交={auction_volume:,.0f}, 前30min均量={avg_vol_pre_auction:,.0f} -> {volume_surprise_factor:.2f}倍")
                        print(f"    - 维度3 (盘口张力): 竞价前买一量={last_snapshot.get('buy_volume1', 0):,.0f}, 卖一量={last_snapshot.get('sell_volume1', 0):,.0f} -> {pre_auction_tension:.4f}")
                        print(f"    - 计算: {auction_price_change:.2f} * log1p({volume_surprise_factor:.2f}) * (1 + {pre_auction_tension:.4f})")
                        print(f"    -> 结果: {results['auction_showdown_score']:.4f}")
                # 分钟降级路径
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
