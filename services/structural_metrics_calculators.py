# services/structural_metrics_calculators.py
import pandas as pd
import numpy as np
from datetime import time
import pandas_ta as ta

class StructuralMetricsCalculators:
    """
    【V25.0 · 计算内核剥离】
    - 核心职责: 封装所有高级结构指标的纯计算逻辑，与服务流程完全解耦。
    - 架构模式: 作为一个无状态的静态工具类，提供一系列独立的计算函数。
    """
    @staticmethod
    def calculate_energy_density_metrics(context: dict) -> dict:
        """
        【V62.0 · 战果量化】
        - 核心升维: 重构 `high_level_consolidation_volume` 的“确证因子”，将刚性的 `np.sign()`
                     升级为基于 `tanh((收盘价 - 阈值) / ATR)` 的平滑函数。此举将对高位博弈
                     成败的评判从“非黑即白”升维至可量化“战果”的连续谱，使指标更精妙圆融。
        - 核心修正: 调和了 `dynamic_reversal_strength` 中动量惩罚因子的计算方式。
        - 核心重构: 秉持“动能同源”原则，统一 `opening_period_thrust` 的计算逻辑。
        """
        group = context['group']
        daily_series_for_day = context['daily_series_for_day']
        atr_14 = context['atr_14']
        tick_df = context.get('tick_df')
        level5_df = context.get('level5_df')
        day_open_qfq = context['day_open_qfq']
        day_high_qfq = context['day_high_qfq']
        day_low_qfq = context['day_low_qfq']
        day_close_qfq = context['day_close_qfq']
        pre_close_qfq = context['pre_close_qfq']
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        results = {
            'intraday_energy_density': np.nan,
            'intraday_thrust_purity': np.nan,
            'volume_burstiness_index': np.nan,
            'auction_impact_score': np.nan,
            'dynamic_reversal_strength': np.nan,
            'reversal_conviction_rate': np.nan,
            'reversal_recovery_rate': np.nan,
            'high_level_consolidation_volume': np.nan,
            'opening_period_thrust': np.nan,
        }
        if pd.notna(atr_14) and atr_14 > 0:
            turnover_rate = pd.to_numeric(daily_series_for_day.get('turnover_rate_f'), errors='coerce')
            if pd.notna(turnover_rate):
                results['intraday_energy_density'] = np.log1p(turnover_rate) / atr_14
        if tick_df is not None and not tick_df.empty:
            total_volume = tick_df['volume'].sum()
            if total_volume > 0:
                if 'price_change' in tick_df.columns and not tick_df['price_change'].isnull().all():
                    self_calculated_change = tick_df['price'].diff().fillna(0)
                    zero_change_mask = tick_df['price_change'] == 0
                    effective_price_change = np.where(zero_change_mask, self_calculated_change, tick_df['price_change'])
                    net_thrust_volume = (tick_df['volume'] * np.sign(effective_price_change)).sum()
                    results['intraday_thrust_purity'] = net_thrust_volume / total_volume
                    if enable_probe and is_target_date:
                        recalculated_count = zero_change_mask.sum()
                        print(f"--- [探针 ASM.{trade_date_str}] intraday_thrust_purity (高频-动能回溯) ---")
                        print(f"    - 节点 (动能回溯): {recalculated_count}/{len(tick_df)} 笔成交触发了价格变动回溯计算。")
                        print(f"    - 原料: 有效净推力成交量={net_thrust_volume:,.0f}, 总成交量={total_volume:,.0f}")
                        print(f"    - 计算: {net_thrust_volume:,.0f} / {total_volume:,.0f}")
                        print(f"    -> 结果: {results['intraday_thrust_purity']:.4f}")
                elif 'type' in tick_df.columns:
                    active_buy_vol = tick_df[tick_df['type'] == 'B']['volume'].sum()
                    active_sell_vol = tick_df[tick_df['type'] == 'S']['volume'].sum()
                    results['intraday_thrust_purity'] = (active_buy_vol - active_sell_vol) / total_volume
        else:
            thrust_vector = (group['close'] - group['open']) * group['vol']
            absolute_energy = abs(group['close'] - group['open']) * group['vol']
            total_energy = absolute_energy.sum()
            if total_energy > 0:
                results['intraday_thrust_purity'] = thrust_vector.sum() / total_energy
        if tick_df is not None and not tick_df.empty:
            results['volume_burstiness_index'] = StructuralMetricsCalculators.calculate_gini(tick_df['volume'].values)
            if enable_probe and is_target_date:
                print(f"--- [探针 ASM.{trade_date_str}] volume_burstiness_index (高频) ---")
                print(f"    - 原料: {len(tick_df)} 笔逐笔成交量序列")
                print(f"    -> 结果: {results['volume_burstiness_index']:.4f}")
        else:
            results['volume_burstiness_index'] = StructuralMetricsCalculators.calculate_gini(group['vol'].values)
        if all(pd.notna(v) for v in [day_open_qfq, pre_close_qfq, atr_14]) and atr_14 > 0:
            gap_magnitude = (day_open_qfq - pre_close_qfq) / atr_14
            if tick_df is not None and not tick_df.empty and level5_df is not None and not level5_df.empty:
                opening_ticks = tick_df[tick_df.index.time < time(9, 35)]
                opening_level5 = level5_df[level5_df.index.time < time(9, 35)]
                if not opening_ticks.empty and not opening_level5.empty:
                    merged_hf = pd.merge_asof(opening_ticks.sort_index(), opening_level5.sort_index(), on='trade_time', direction='backward')
                    merged_hf['mid_price'] = (merged_hf['buy_price1'] + merged_hf['sell_price1']) / 2
                    merged_hf['prev_mid_price'] = merged_hf['mid_price'].shift(1)
                    buy_pressure = np.where(merged_hf['mid_price'] >= merged_hf['prev_mid_price'], merged_hf['buy_volume1'].shift(1), 0)
                    sell_pressure = np.where(merged_hf['mid_price'] <= merged_hf['prev_mid_price'], merged_hf['sell_volume1'].shift(1), 0)
                    merged_hf['ofi'] = buy_pressure - sell_pressure
                    opening_ofi = merged_hf['ofi'].sum()
                    opening_volume = merged_hf['volume'].sum()
                    if opening_volume > 0:
                        conviction_factor = np.tanh(opening_ofi / opening_volume)
                        results['auction_impact_score'] = gap_magnitude * (1 + conviction_factor * np.sign(gap_magnitude))
                        if enable_probe and is_target_date:
                            print(f"--- [探针 ASM.{trade_date_str}] auction_impact_score (高频) ---")
                            print(f"    - 原料: 开盘价={day_open_qfq:.2f}, 昨收={pre_close_qfq:.2f}, ATR={atr_14:.4f}")
                            print(f"    - 节点1 (缺口): ({day_open_qfq:.2f} - {pre_close_qfq:.2f}) / {atr_14:.4f} = {gap_magnitude:.4f}")
                            print(f"    - 原料2: 开盘5分钟OFI={opening_ofi:,.0f}, 成交量={opening_volume:,.0f}")
                            print(f"    - 节点2 (信念): tanh({opening_ofi:,.0f} / {opening_volume:,.0f}) = {conviction_factor:.4f}")
                            aligned_conviction = conviction_factor * np.sign(gap_magnitude)
                            print(f"    - 节点3 (同调信念): {conviction_factor:.4f} * sign({gap_magnitude:.4f}) = {aligned_conviction:.4f}")
                            print(f"    - 计算: {gap_magnitude:.4f} * (1 + {aligned_conviction:.4f})")
                            print(f"    -> 结果: {results['auction_impact_score']:.4f}")
                    else:
                        results['auction_impact_score'] = gap_magnitude
                else:
                    results['auction_impact_score'] = gap_magnitude
            else:
                results['auction_impact_score'] = gap_magnitude
        try:
            from scipy.signal import find_peaks
            prominence_source = "静态回退"
            if pd.notna(atr_14) and atr_14 > 0:
                dynamic_prominence = atr_14 * 0.05
                prominence_source = f"动态ATR({atr_14:.2f}*5%)"
            else:
                dynamic_prominence = 0.01
            peaks, _ = find_peaks(group['high'], distance=5, prominence=dynamic_prominence)
            troughs, _ = find_peaks(-group['low'], distance=5, prominence=dynamic_prominence)
            if len(troughs) > 0 and len(peaks) > 0:
                reversal_details = []
                all_extrema = sorted(np.concatenate([peaks, troughs]))
                first_trough_idx = -1
                for i, extremum_pos in enumerate(all_extrema):
                    if extremum_pos in troughs:
                        first_trough_idx = i
                        break
                if first_trough_idx != -1:
                    for i in range(first_trough_idx, len(all_extrema) - 1):
                        if all_extrema[i] in troughs and all_extrema[i+1] in peaks:
                            trough_pos = all_extrema[i]
                            peak_pos = all_extrema[i+1]
                            prev_peak_candidates = peaks[peaks < trough_pos]
                            if len(prev_peak_candidates) > 0:
                                prev_peak_pos = prev_peak_candidates[-1]
                                falling_phase = group.iloc[prev_peak_pos:trough_pos+1]
                                rebounding_phase = group.iloc[trough_pos:peak_pos+1]
                                vol_fall = falling_phase['vol'].sum()
                                vol_rebound = rebounding_phase['vol'].sum()
                                if not falling_phase.empty and not rebounding_phase.empty and \
                                   vol_fall > 0 and vol_rebound > 0:
                                    vwap_fall = falling_phase['amount'].sum() / vol_fall
                                    vwap_rebound = rebounding_phase['amount'].sum() / vol_rebound
                                    if vwap_fall > 0:
                                        price_momentum = (vwap_rebound / vwap_fall - 1)
                                        fall_magnitude = group.iloc[prev_peak_pos]['high'] - group.iloc[trough_pos]['low']
                                        rebound_magnitude = group.iloc[peak_pos]['high'] - group.iloc[trough_pos]['low']
                                        recovery_rate = rebound_magnitude / fall_magnitude if fall_magnitude > 0 else 0
                                        if recovery_rate > 1:
                                            volume_factor = np.log1p(vol_rebound / vol_fall)
                                        else:
                                            volume_factor = np.log1p(vol_fall / vol_rebound)
                                        momentum = (price_momentum * volume_factor) * 100
                                        reversal_details.append({
                                            "momentum": momentum,
                                            "fall_magnitude": fall_magnitude,
                                            "rebound_magnitude": rebound_magnitude
                                        })
                if reversal_details:
                    positive_momentums = [r['momentum'] for r in reversal_details if r['momentum'] > 0]
                    negative_momentums = [r['momentum'] for r in reversal_details if r['momentum'] <= 0]
                    sum_positive_momentum = np.sum(positive_momentums)
                    sum_abs_negative_momentum = np.sum(np.abs(negative_momentums))
                    total_abs_momentum = sum_positive_momentum + sum_abs_negative_momentum
                    conviction_rate = 0.0
                    if total_abs_momentum > 0:
                        conviction_rate = sum_positive_momentum / total_abs_momentum
                    results['reversal_conviction_rate'] = conviction_rate
                    if positive_momentums:
                        raw_strength = np.mean(positive_momentums)
                        final_strength = raw_strength * conviction_rate
                        results['dynamic_reversal_strength'] = final_strength
                        successful_reversals = [r for r in reversal_details if r['momentum'] > 0]
                        recovery_ratios = [
                            r['rebound_magnitude'] / r['fall_magnitude']
                            for r in successful_reversals if r['fall_magnitude'] > 0
                        ]
                        if recovery_ratios:
                            results['reversal_recovery_rate'] = np.mean(recovery_ratios)
        except ImportError:
            pass
        price_range = day_high_qfq - day_low_qfq
        # 修改代码块：升维 high_level_consolidation_volume 的计算逻辑
        if price_range > 0 and pd.notna(atr_14) and atr_14 > 0:
            high_level_threshold = day_high_qfq - 0.25 * price_range
            volume_ratio = 0.0
            if tick_df is not None and not tick_df.empty:
                high_vol = tick_df[tick_df['price'] >= high_level_threshold]['volume'].sum()
                total_vol = tick_df['volume'].sum()
                if total_vol > 0:
                    volume_ratio = high_vol / total_vol
            else:
                total_volume = group['vol'].sum()
                if total_volume > 0:
                    volume_ratio = group[group['high'] >= high_level_threshold]['vol'].sum() / total_volume
            # 升维确证因子：从 sign() 升级为 tanh(ATR标准化距离)
            distance_from_threshold = day_close_qfq - high_level_threshold
            normalized_distance = distance_from_threshold / atr_14
            confirmation_factor = np.tanh(normalized_distance)
            results['high_level_consolidation_volume'] = volume_ratio * confirmation_factor
            if enable_probe and is_target_date:
                print(f"--- [探针 ASM.{trade_date_str}] high_level_consolidation_volume (战果量化) ---")
                print(f"    - 原料: 高位阈值={high_level_threshold:.2f}, 收盘价={day_close_qfq:.2f}, 高位成交量占比={volume_ratio:.4f}, ATR={atr_14:.4f}")
                print(f"    - 节点1 (战果): ({day_close_qfq:.2f} - {high_level_threshold:.2f}) / {atr_14:.4f} = {normalized_distance:.4f}")
                print(f"    - 节点2 (确证因子): tanh({normalized_distance:.4f}) = {confirmation_factor:.4f}")
                print(f"    - 计算: {volume_ratio:.4f} * {confirmation_factor:.4f}")
                print(f"    -> 结果: {results['high_level_consolidation_volume']:.4f}")
        if tick_df is not None and not tick_df.empty:
            opening_ticks = tick_df.between_time('09:30:00', '09:59:59')
            if not opening_ticks.empty:
                opening_total_vol = opening_ticks['volume'].sum()
                if opening_total_vol > 0:
                    if 'price_change' in opening_ticks.columns and not opening_ticks['price_change'].isnull().all():
                        self_calculated_change = opening_ticks['price'].diff().fillna(0)
                        zero_change_mask = opening_ticks['price_change'] == 0
                        effective_price_change = np.where(zero_change_mask, self_calculated_change, opening_ticks['price_change'])
                        net_opening_thrust_volume = (opening_ticks['volume'] * np.sign(effective_price_change)).sum()
                        results['opening_period_thrust'] = net_opening_thrust_volume / opening_total_vol
                    elif 'type' in opening_ticks.columns:
                        opening_buy_vol = opening_ticks[opening_ticks['type'] == 'B']['volume'].sum()
                        opening_sell_vol = opening_ticks[opening_ticks['type'] == 'S']['volume'].sum()
                        results['opening_period_thrust'] = (opening_buy_vol - opening_sell_vol) / opening_total_vol
        return results

    @staticmethod
    def calculate_control_metrics(context: dict) -> dict:
        """
        【V63.0 · 劲力合一】
        - 核心升维: 全面重构“脉冲”、“加速”与“沉寂”三大指标，从度量“力”的大小与方向，
                     升维至度量“劲”的效率与效果，更深层次地洞察控盘行为的性价比与真实意图。
        - `opening_impulse_efficiency` 新增: 衡量开盘期“单位成交量撬动的价格变动”，即“四两拨千斤”的效率。
        - `midday_narrow_range_gravity` 新增: 直接计算盘中相对活跃时段的波动率收缩度，精准刻画“沉寂蓄势”。
        - `tail_acceleration_efficiency` 新增: 衡量尾盘加速的真实“性价比”，辨别“高效偷袭”与“力竭冲高”。
        """
        group = context['group']
        continuous_group = context['continuous_group']
        tick_df = context.get('tick_df')
        daily_info = context['daily_series_for_day']
        day_open_qfq = context['day_open_qfq']
        day_close_qfq = context['day_close_qfq']
        atr_14 = context['atr_14']
        total_volume_safe = context['total_volume_safe']
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        results = {}
        if group.empty or total_volume_safe == 0 or not pd.notna(atr_14) or atr_14 == 0:
            return {}
        dispersion_raw = np.nan
        if tick_df is not None and not tick_df.empty:
            weighted_price_mean = np.average(tick_df['price'], weights=tick_df['volume'])
            variance = np.average((tick_df['price'] - weighted_price_mean)**2, weights=tick_df['volume'])
            dispersion_raw = np.sqrt(variance)
            results['cost_dispersion_index'] = dispersion_raw / atr_14
        else:
            weighted_price_mean = np.average(group['close'], weights=group['vol'])
            variance = np.average((group['close'] - weighted_price_mean)**2, weights=group['vol'])
            dispersion_raw = np.sqrt(variance)
            results['cost_dispersion_index'] = dispersion_raw / atr_14
        first_half = continuous_group[continuous_group.index.time < time(11, 30)]
        second_half = continuous_group[continuous_group.index.time >= time(13, 0)]
        if not first_half.empty and not second_half.empty and first_half['vol'].sum() > 0 and second_half['vol'].sum() > 0:
            vwap_first = (first_half['amount'].sum() / first_half['vol'].sum())
            vwap_second = (second_half['amount'].sum() / second_half['vol'].sum())
            results['intraday_pnl_imbalance'] = (vwap_second - vwap_first) / atr_14
        if 'minute_vwap' in continuous_group.columns and not continuous_group['minute_vwap'].isnull().all():
            if tick_df is not None and not tick_df.empty:
                continuous_group.index.name = 'trade_time'
                merged_df = pd.merge_asof(tick_df.sort_index(), continuous_group[['minute_vwap']].sort_index(), on='trade_time', direction='backward')
                merged_df['position'] = np.sign(merged_df['price'] - merged_df['minute_vwap'])
                crossings = (merged_df['position'].diff().abs() == 2).sum()
                results['mean_reversion_frequency'] = (crossings / len(tick_df)) * 1000 if len(tick_df) > 0 else 0
            else:
                position = np.sign(continuous_group['close'] - continuous_group['minute_vwap'])
                crossings = (position.diff().abs() == 2).sum()
                results['mean_reversion_frequency'] = (crossings / len(continuous_group)) * 100 if len(continuous_group) > 0 else 0
        price_change = day_close_qfq - day_open_qfq
        sum_abs_minute_change = (continuous_group['high'] - continuous_group['low']).sum()
        if sum_abs_minute_change > 0:
            er_raw = abs(price_change) / sum_abs_minute_change
            thrust_purity = context.get('intraday_thrust_purity', 0)
            results['trend_efficiency_ratio'] = er_raw * (1 + thrust_purity) * np.sign(price_change)
        minute_return = continuous_group['close'].pct_change().fillna(0)
        minute_volume = continuous_group['vol']
        advancing_mask = continuous_group['close'] > continuous_group['open']
        declining_mask = continuous_group['close'] < continuous_group['open']
        if advancing_mask.sum() > 2 and declining_mask.sum() > 2:
            corr_adv = minute_return[advancing_mask].corr(minute_volume[advancing_mask])
            corr_dec = minute_return[declining_mask].corr(minute_volume[declining_mask])
            corr_adv = corr_adv if pd.notna(corr_adv) else 0
            corr_dec = corr_dec if pd.notna(corr_dec) else 0
            trend_direction = np.sign(day_close_qfq - day_open_qfq) if (day_close_qfq != day_open_qfq) else 1
            results['pullback_depth_ratio'] = (corr_adv - corr_dec) * trend_direction
        # 划分日内三个核心时段
        open_period_df = continuous_group.between_time('09:30', '10:00')
        mid_period_df = continuous_group.between_time('10:01', '14:29')
        tail_period_df = continuous_group.between_time('14:30', '15:00')
        # 修改代码块：升维为 opening_impulse_efficiency
        if not open_period_df.empty and total_volume_safe > 0:
            open_vol_ratio = open_period_df['vol'].sum() / total_volume_safe
            if open_vol_ratio > 0:
                open_price_change = open_period_df['close'].iloc[-1] - open_period_df['open'].iloc[0]
                price_change_norm = open_price_change / atr_14
                results['opening_impulse_efficiency'] = price_change_norm / open_vol_ratio
                if enable_probe and is_target_date:
                    print(f"--- [探针 ASM.{trade_date_str}] opening_impulse_efficiency (控盘-劲力合一) ---")
                    print(f"    - 原料: 开盘期价格变动={open_price_change:.2f}, 成交量占比={open_vol_ratio:.4f}, ATR={atr_14:.4f}")
                    print(f"    - 节点 (标准化价格变动): {open_price_change:.2f} / {atr_14:.4f} = {price_change_norm:.4f}")
                    print(f"    - 计算 (效率): {price_change_norm:.4f} / {open_vol_ratio:.4f}")
                    print(f"    -> 结果: {results.get('opening_impulse_efficiency', np.nan):.4f}")
        # 修改代码块：升维为 midday_narrow_range_gravity
        if not mid_period_df.empty and not open_period_df.empty and not tail_period_df.empty:
            volatility_mid = mid_period_df['close'].pct_change().std()
            active_period_df = pd.concat([open_period_df, tail_period_df])
            volatility_active = active_period_df['close'].pct_change().std()
            if pd.notna(volatility_mid) and pd.notna(volatility_active) and volatility_active > 0:
                results['midday_narrow_range_gravity'] = 1 - (volatility_mid / volatility_active)
                if enable_probe and is_target_date:
                    print(f"--- [探针 ASM.{trade_date_str}] midday_narrow_range_gravity (控盘-劲力合一) ---")
                    print(f"    - 原料: 盘中波动率={volatility_mid:.6f}, 活跃时段波动率={volatility_active:.6f}")
                    print(f"    - 计算 (引力): 1 - ({volatility_mid:.6f} / {volatility_active:.6f})")
                    print(f"    -> 结果: {results.get('midday_narrow_range_gravity', np.nan):.4f}")
        # 修改代码块：升维为 tail_acceleration_efficiency
        if not tail_period_df.empty and total_volume_safe > 0:
            tail_vol_ratio = tail_period_df['vol'].sum() / total_volume_safe
            if tail_vol_ratio > 0:
                tail_price_change = tail_period_df['close'].iloc[-1] - tail_period_df['open'].iloc[0]
                price_change_norm = tail_price_change / atr_14
                results['tail_acceleration_efficiency'] = price_change_norm / tail_vol_ratio
                if enable_probe and is_target_date:
                    print(f"--- [探针 ASM.{trade_date_str}] tail_acceleration_efficiency (控盘-劲力合一) ---")
                    print(f"    - 原料: 尾盘期价格变动={tail_price_change:.2f}, 成交量占比={tail_vol_ratio:.4f}, ATR={atr_14:.4f}")
                    print(f"    - 节点 (标准化价格变动): {tail_price_change:.2f} / {atr_14:.4f} = {price_change_norm:.4f}")
                    print(f"    - 计算 (效率): {price_change_norm:.4f} / {tail_vol_ratio:.4f}")
                    print(f"    -> 结果: {results.get('tail_acceleration_efficiency', np.nan):.4f}")
        if not tail_period_df.empty and not mid_period_df.empty and mid_period_df['vol'].mean() > 0:
            accel_ratio = tail_period_df['vol'].mean() / mid_period_df['vol'].mean()
            tail_thrust_purity = np.nan
            if tick_df is not None:
                tail_ticks = tick_df[tick_df.index.time >= time(14, 30)]
                if not tail_ticks.empty and tail_ticks['volume'].sum() > 0:
                    tail_total_vol = tail_ticks['volume'].sum()
                    if 'price_change' in tail_ticks.columns and not tail_ticks['price_change'].isnull().all():
                        self_calculated_change = tail_ticks['price'].diff().fillna(0)
                        zero_change_mask = tail_ticks['price_change'] == 0
                        effective_price_change = np.where(zero_change_mask, self_calculated_change, tail_ticks['price_change'])
                        net_thrust_vol = (tail_ticks['volume'] * np.sign(effective_price_change)).sum()
                        tail_thrust_purity = net_thrust_vol / tail_total_vol
                    elif 'type' in tail_ticks.columns:
                        buy_vol = tail_ticks[tail_ticks['type'] == 'B']['volume'].sum()
                        sell_vol = tail_ticks[tail_ticks['type'] == 'S']['volume'].sum()
                        tail_thrust_purity = (buy_vol - sell_vol) / tail_total_vol
            vpoc = context.get('_today_vpoc', np.nan)
            if pd.notna(vpoc):
                deviation_magnitude = (day_close_qfq - vpoc) / atr_14
                tail_force_factor = np.log1p(accel_ratio)
                conviction_purity = tail_thrust_purity if pd.notna(tail_thrust_purity) else np.sign(day_close_qfq - vpoc)
                results['closing_conviction_score'] = deviation_magnitude * tail_force_factor * conviction_purity
        return results

    @staticmethod
    def calculate_game_efficiency_metrics(context: dict) -> dict:
        """
        【V65.0 · 博弈穿透】
        - 核心升维: 全面重构博弈效率指标，引入“冲击成本”和“路径效率”概念。
        - `breakthrough/defense_cost_index` 新增: 基于高频Tick，精确计算多空双方在推动价格时
                     付出的平均“滑点成本”，量化突破与防御的真实代价。
        - `trend_asymmetry_index` 新增: 比较上涨与下跌分钟的“趋势效率”，从“幅度”不对称性
                     的度量，升维至“路径”顺畅度的不对称性分析。
        """
        group = context['group']
        tick_df = context.get('tick_df')
        day_open_qfq = context['day_open_qfq']
        day_close_qfq = context['day_close_qfq']
        atr_14 = context['atr_14']
        intraday_thrust_purity = context.get('intraday_thrust_purity')
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        results = {}
        if group.empty:
            return results
        # 1. 升维：趋势不对称指数 (Trend Asymmetry Index) - 分钟级
        up_minutes = group[group['close'] > group['open']]
        down_minutes = group[group['close'] < group['open']]
        if not up_minutes.empty and not down_minutes.empty:
            up_path = (up_minutes['high'] - up_minutes['low']).sum()
            up_net_change = (up_minutes['close'] - up_minutes['open']).sum()
            down_path = (down_minutes['high'] - down_minutes['low']).sum()
            down_net_change = abs(down_minutes['close'] - down_minutes['open']).sum()
            eff_up = up_net_change / up_path if up_path > 0 else 0
            eff_down = down_net_change / down_path if down_path > 0 else 0
            if eff_up > 0 and eff_down > 0:
                results['trend_asymmetry_index'] = np.log(eff_up / eff_down)
                if enable_probe and is_target_date:
                    print(f"--- [探针 ASM.{trade_date_str}] trend_asymmetry_index (博弈) ---")
                    print(f"    - 原料: 上涨分钟数={len(up_minutes)}, 下跌分钟数={len(down_minutes)}")
                    print(f"    - 节点: 上涨效率={eff_up:.4f}, 下跌效率={eff_down:.4f}")
                    print(f"    - 计算: log({eff_up:.4f} / {eff_down:.4f})")
                    print(f"    -> 结果: {results.get('trend_asymmetry_index', np.nan):.4f}")
        # 2. 保留：推力效能分 (Thrust Efficiency Score)
        if all(pd.notna(v) for v in [day_close_qfq, day_open_qfq, atr_14, intraday_thrust_purity]) and atr_14 > 0:
            price_change_in_atr = (day_close_qfq - day_open_qfq) / atr_14
            effort_factor = 1 - abs(intraday_thrust_purity) + 1e-9
            results['thrust_efficiency_score'] = price_change_in_atr / effort_factor
        # 3. 升维：突破/防御成本指数 (Breakthrough/Defense Cost Index) - 高频
        if tick_df is not None and not tick_df.empty and len(tick_df) > 1 and pd.notna(atr_14) and atr_14 > 0:
            tick_df['prev_price'] = tick_df['price'].shift(1)
            tick_df['price_diff'] = tick_df['price'] - tick_df['prev_price']
            # 突破成本 (多方)
            up_thrust_ticks = tick_df[tick_df['price_diff'] > 0]
            if not up_thrust_ticks.empty:
                total_up_vol = up_thrust_ticks['volume'].sum()
                if total_up_vol > 0:
                    weighted_avg_slippage_up = np.average(up_thrust_ticks['price_diff'], weights=up_thrust_ticks['volume'])
                    results['breakthrough_cost_index'] = weighted_avg_slippage_up / atr_14
                    if enable_probe and is_target_date:
                        print(f"--- [探针 ASM.{trade_date_str}] breakthrough_cost_index (博弈) ---")
                        print(f"    - 原料: {len(up_thrust_ticks)}笔上涨Tick, 总成交量={total_up_vol:,.0f}, ATR={atr_14:.4f}")
                        print(f"    - 节点 (成交量加权平均滑点): {weighted_avg_slippage_up:.6f}")
                        print(f"    - 计算: {weighted_avg_slippage_up:.6f} / {atr_14:.4f}")
                        print(f"    -> 结果: {results.get('breakthrough_cost_index', np.nan):.4f}")
            # 防御成本 (空方)
            down_thrust_ticks = tick_df[tick_df['price_diff'] < 0]
            if not down_thrust_ticks.empty:
                total_down_vol = down_thrust_ticks['volume'].sum()
                if total_down_vol > 0:
                    weighted_avg_slippage_down = np.average(abs(down_thrust_ticks['price_diff']), weights=down_thrust_ticks['volume'])
                    results['defense_cost_index'] = weighted_avg_slippage_down / atr_14
                    if enable_probe and is_target_date:
                        print(f"--- [探针 ASM.{trade_date_str}] defense_cost_index (博弈) ---")
                        print(f"    - 原料: {len(down_thrust_ticks)}笔下跌Tick, 总成交量={total_down_vol:,.0f}, ATR={atr_14:.4f}")
                        print(f"    - 节点 (成交量加权平均滑点): {weighted_avg_slippage_down:.6f}")
                        print(f"    - 计算: {weighted_avg_slippage_down:.6f} / {atr_14:.4f}")
                        print(f"    -> 结果: {results.get('defense_cost_index', np.nan):.4f}")
        return results

    @staticmethod
    def calculate_gini(array: np.ndarray) -> float:
        """
        【V22.0 · 计算内核静态化】
        计算基尼系数
        """
        if array is None or len(array) < 2 or np.sum(array) == 0:
            return 0.0
        sorted_array = np.sort(array)
        n = len(array)
        cum_array = np.cumsum(sorted_array, dtype=float)
        return (n + 1 - 2 * np.sum(cum_array) / cum_array[-1]) / n












