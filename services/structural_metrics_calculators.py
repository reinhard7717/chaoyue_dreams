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
        【V41.0 · 动能同源】
        - 核心重构: 秉持“动能同源”原则，将 `opening_period_thrust` 指标的计算逻辑，
                     从原始的 B/S 盘意图推断，全面升级为与 `intraday_thrust_purity`
                     一致的“动能回溯”方法。确保了体系内所有“推力”度量衡的一致性与高精度。
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
                    if enable_probe and is_target_date:
                        print(f"--- [探针 ASM.{trade_date_str}] intraday_thrust_purity (高频-基于B/S盘) ---")
                        print(f"    - 原料: 总成交量={total_volume:,.0f}, 主动买量={active_buy_vol:,.0f}, 主动卖量={active_sell_vol:,.0f}")
                        print(f"    - 计算: ({active_buy_vol:,.0f} - {active_sell_vol:,.0f}) / {total_volume:,.0f}")
                        print(f"    -> 结果: {results['intraday_thrust_purity']:.4f}")
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
                                            volume_factor = vol_fall / vol_rebound
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
                    if enable_probe and is_target_date:
                        print(f"--- [探针 ASM.{trade_date_str}] reversal_conviction_rate (分钟) ---")
                        print(f"    - 前置: 使用 {prominence_source} 显著性阈值 = {dynamic_prominence:.4f}")
                        print(f"    - 原料: 成功动能列表({len(positive_momentums)}次), 失败动能列表({len(negative_momentums)}次)")
                        print(f"    - 节点1 (总成功动能): sum({[f'{m:.2f}' for m in positive_momentums]}) = {sum_positive_momentum:.4f}")
                        print(f"    - 节点2 (总失败惩罚): sum(abs({[f'{m:.2f}' for m in negative_momentums]})) = {sum_abs_negative_momentum:.4f}")
                        print(f"    - 计算 (动能净比率): {sum_positive_momentum:.4f} / ({sum_positive_momentum:.4f} + {sum_abs_negative_momentum:.4f})")
                        print(f"    -> 结果: {conviction_rate:.4f}")
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
                            if enable_probe and is_target_date:
                                print(f"--- [探针 ASM.{trade_date_str}] reversal_recovery_rate (分钟) ---")
                                print(f"    - 原料: {len(recovery_ratios)} 次成功反转的收复率")
                                print(f"    - 节点 (收复率序列): {[f'{r:.2f}' for r in recovery_ratios]}")
                                print(f"    - 计算: mean of recovery ratios")
                                print(f"    -> 结果: {results['reversal_recovery_rate']:.4f}")
                        if enable_probe and is_target_date:
                            print(f"--- [探针 ASM.{trade_date_str}] dynamic_reversal_strength (分钟) ---")
                            print(f"    - 前置: 识别出 {len(reversal_details)} 次尝试, {len(positive_momentums)} 次成功")
                            print(f"    - 节点1 (原始平均强度): mean({[f'{m:.2f}' for m in positive_momentums]}) = {raw_strength:.4f}")
                            print(f"    - 节点2 (信念权重): {conviction_rate:.4f}")
                            print(f"    - 计算 (信念加权): {raw_strength:.4f} * {conviction_rate:.4f}")
                            print(f"    -> 结果: {final_strength:.4f}")
        except ImportError:
            pass
        price_range = day_high_qfq - day_low_qfq
        if price_range > 0:
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
            confirmation_factor = np.sign(day_close_qfq - high_level_threshold)
            results['high_level_consolidation_volume'] = volume_ratio * confirmation_factor
            if enable_probe and is_target_date:
                print(f"--- [探针 ASM.{trade_date_str}] high_level_consolidation_volume (高频) ---")
                print(f"    - 原料: 高位阈值={high_level_threshold:.2f}, 收盘价={day_close_qfq:.2f}, 高位成交量占比={volume_ratio:.4f}")
                print(f"    - 节点 (确证因子): sign({day_close_qfq:.2f} - {high_level_threshold:.2f}) = {confirmation_factor:.0f}")
                print(f"    - 计算: {volume_ratio:.4f} * {confirmation_factor:.0f}")
                print(f"    -> 结果: {results['high_level_consolidation_volume']:.4f}")
        if tick_df is not None and not tick_df.empty: # 重构 opening_period_thrust 计算逻辑
            opening_ticks = tick_df.between_time('09:30:00', '09:59:59')
            if not opening_ticks.empty:
                opening_total_vol = opening_ticks['volume'].sum()
                if opening_total_vol > 0:
                    # 优先使用基于 price_change 的精确动能计算
                    if 'price_change' in opening_ticks.columns and not opening_ticks['price_change'].isnull().all():
                        self_calculated_change = opening_ticks['price'].diff().fillna(0)
                        zero_change_mask = opening_ticks['price_change'] == 0
                        effective_price_change = np.where(zero_change_mask, self_calculated_change, opening_ticks['price_change'])
                        net_opening_thrust_volume = (opening_ticks['volume'] * np.sign(effective_price_change)).sum()
                        results['opening_period_thrust'] = net_opening_thrust_volume / opening_total_vol
                        if enable_probe and is_target_date:
                            recalculated_count = zero_change_mask.sum()
                            print(f"--- [探针 ASM.{trade_date_str}] opening_period_thrust (高频-动能回溯) ---")
                            print(f"    - 节点 (动能回溯): {recalculated_count}/{len(opening_ticks)} 笔成交触发了价格变动回溯计算。")
                            print(f"    - 原料: 开盘期净推力成交量={net_opening_thrust_volume:,.0f}, 开盘期总量={opening_total_vol:,.0f}")
                            print(f"    - 计算: {net_opening_thrust_volume:,.0f} / {opening_total_vol:,.0f}")
                            print(f"    -> 结果: {results['opening_period_thrust']:.4f}")
                    # 回退到基于 B/S 盘的传统计算
                    elif 'type' in opening_ticks.columns:
                        opening_buy_vol = opening_ticks[opening_ticks['type'] == 'B']['volume'].sum()
                        opening_sell_vol = opening_ticks[opening_ticks['type'] == 'S']['volume'].sum()
                        results['opening_period_thrust'] = (opening_buy_vol - opening_sell_vol) / opening_total_vol
                        if enable_probe and is_target_date:
                            print(f"--- [探针 ASM.{trade_date_str}] opening_period_thrust (高频-基于B/S盘) ---")
                            print(f"    - 原料: 开盘买量={opening_buy_vol:,.0f}, 开盘卖量={opening_sell_vol:,.0f}, 开盘总量={opening_total_vol:,.0f}")
                            print(f"    - 计算: ({opening_buy_vol:,.0f} - {opening_sell_vol:,.0f}) / {opening_total_vol:,.0f}")
                            print(f"    -> 结果: {results['opening_period_thrust']:.4f}")
        return results

    @staticmethod
    def calculate_control_metrics(context: dict) -> dict:
        """
        【V52.0 · 动能归一】
        - 核心修正: 统一了 `opening_volume_impulse` 指标内部的“推力纯度”计算逻辑。
                     将其从一个简化的旧版本，全面升级为系统标准的“动能回溯”算法，
                     与 `opening_period_thrust` 等指标的计算方式完全看齐，
                     根除了“动能不同源”的逻辑瑕疵，确保了度量衡的一致性。
        - 探针升级: 升级 `closing_conviction_score` 的探针，使其明确打印出所引用的VPOC值，
                     增强调试的透明度，确保计算逻辑的每一个环节都清晰可追溯。
        """
        group = context['group']
        continuous_group = context['continuous_group']
        tick_df = context.get('tick_df')
        level5_df = context.get('level5')
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
        if enable_probe and is_target_date:
            print(f"--- [探针 ASM.{trade_date_str}] cost_dispersion_index (控盘) ---")
            print(f"    - 原料: {'Tick' if tick_df is not None else '分钟'}数据, ATR={atr_14:.4f}")
            print(f"    - 节点 (成交加权价格标准差): {dispersion_raw:.4f}")
            print(f"    - 计算: {dispersion_raw:.4f} / {atr_14:.4f}")
            print(f"    -> 结果: {results.get('cost_dispersion_index', np.nan):.4f}")
        first_half = continuous_group[continuous_group.index.time < time(11, 30)]
        second_half = continuous_group[continuous_group.index.time >= time(13, 0)]
        if not first_half.empty and not second_half.empty and first_half['vol'].sum() > 0 and second_half['vol'].sum() > 0:
            vwap_first = (first_half['amount'].sum() / first_half['vol'].sum())
            vwap_second = (second_half['amount'].sum() / second_half['vol'].sum())
            results['intraday_pnl_imbalance'] = (vwap_second - vwap_first) / atr_14
            if enable_probe and is_target_date:
                print(f"--- [探针 ASM.{trade_date_str}] intraday_pnl_imbalance (控盘) ---")
                print(f"    - 原料: 上半场VWAP={vwap_first:.3f}, 下半场VWAP={vwap_second:.3f}, ATR={atr_14:.4f}")
                print(f"    - 计算: ({vwap_second:.3f} - {vwap_first:.3f}) / {atr_14:.4f}")
                print(f"    -> 结果: {results.get('intraday_pnl_imbalance', np.nan):.4f}")
        if 'minute_vwap' in continuous_group.columns:
            deviation = (continuous_group['close'] - continuous_group['minute_vwap'])
            weighted_deviation = (deviation.abs() * continuous_group['vol']).sum() / total_volume_safe
            results['mean_reversion_frequency'] = weighted_deviation / atr_14
            if enable_probe and is_target_date:
                print(f"--- [探针 ASM.{trade_date_str}] mean_reversion_frequency (控盘-VWAP引力) ---")
                print(f"    - 原料: 分钟收盘价与VWAP序列, ATR={atr_14:.4f}")
                print(f"    - 节点 (成交加权平均绝对偏离): {weighted_deviation:.4f}")
                print(f"    - 计算: {weighted_deviation:.4f} / {atr_14:.4f}")
                print(f"    -> 结果: {results.get('mean_reversion_frequency', np.nan):.4f}")
        price_change = day_close_qfq - day_open_qfq
        sum_abs_minute_change = (continuous_group['high'] - continuous_group['low']).sum()
        if sum_abs_minute_change > 0:
            er_raw = price_change / sum_abs_minute_change
            avg_daily_vol_5 = daily_info.get('VMA_5', total_volume_safe)
            vol_factor = np.log1p(total_volume_safe / avg_daily_vol_5) if avg_daily_vol_5 > 0 else 1
            results['trend_efficiency_ratio'] = er_raw * vol_factor
            if enable_probe and is_target_date:
                print(f"--- [探针 ASM.{trade_date_str}] trend_efficiency_ratio (控盘) ---")
                print(f"    - 原料: 价格净变动={price_change:.2f}, 累计振幅={sum_abs_minute_change:.2f}, 成交量因子={vol_factor:.4f}")
                print(f"    - 节点 (原始效率): {er_raw:.4f}")
                print(f"    - 计算: {er_raw:.4f} * {vol_factor:.4f}")
                print(f"    -> 结果: {results.get('trend_efficiency_ratio', np.nan):.4f}")
        minute_return = continuous_group['close'].pct_change().fillna(0)
        minute_volume = continuous_group['vol']
        if len(minute_return) > 2:
            corr = minute_return.corr(minute_volume)
            results['pullback_depth_ratio'] = corr
            if enable_probe and is_target_date:
                print(f"--- [探针 ASM.{trade_date_str}] pullback_depth_ratio (控盘-量价相关性) ---")
                print(f"    - 原料: 分钟收益率序列, 分钟成交量序列")
                print(f"    - 计算: corr(returns, volumes)")
                print(f"    -> 结果: {results.get('pullback_depth_ratio', np.nan):.4f}")
        open_period_df = continuous_group[continuous_group.index.time < time(9, 45)]
        mid_period_df = continuous_group[(continuous_group.index.time >= time(10, 30)) & (continuous_group.index.time < time(14, 30))]
        tail_period_df = continuous_group[continuous_group.index.time >= time(14, 30)]
        if not open_period_df.empty:
            open_vol = open_period_df['vol'].sum()
            avg_min_vol = total_volume_safe / 240
            vol_impulse_ratio = (open_vol / 15) / avg_min_vol if avg_min_vol > 0 else 1.0
            thrust_purity = np.nan
            # 修改代码块：全面升级为标准的“动能回溯”算法
            if tick_df is not None:
                open_ticks = tick_df[tick_df.index.time < time(9, 45)]
                if not open_ticks.empty and open_ticks['volume'].sum() > 0:
                    open_total_vol = open_ticks['volume'].sum()
                    if 'price_change' in open_ticks.columns and not open_ticks['price_change'].isnull().all():
                        self_calculated_change = open_ticks['price'].diff().fillna(0)
                        zero_change_mask = open_ticks['price_change'] == 0
                        effective_price_change = np.where(zero_change_mask, self_calculated_change, open_ticks['price_change'])
                        net_thrust_vol = (open_ticks['volume'] * np.sign(effective_price_change)).sum()
                        thrust_purity = net_thrust_vol / open_total_vol
                    elif 'type' in open_ticks.columns:
                        buy_vol = open_ticks[open_ticks['type'] == 'B']['volume'].sum()
                        sell_vol = open_ticks[open_ticks['type'] == 'S']['volume'].sum()
                        thrust_purity = (buy_vol - sell_vol) / open_total_vol
            if pd.notna(thrust_purity):
                results['opening_volume_impulse'] = vol_impulse_ratio * thrust_purity
            else:
                results['opening_volume_impulse'] = vol_impulse_ratio * np.sign(open_period_df['close'].iloc[-1] - open_period_df['open'].iloc[0])
            if enable_probe and is_target_date:
                print(f"--- [探针 ASM.{trade_date_str}] opening_volume_impulse (控盘-动能归一) ---")
                print(f"    - 原料: 开盘15分钟成交均值/全天均值={vol_impulse_ratio:.2f}, 开盘期推力纯度={thrust_purity:.4f}")
                print(f"    - 计算: {vol_impulse_ratio:.2f} * {thrust_purity:.4f}")
                print(f"    -> 结果: {results.get('opening_volume_impulse', np.nan):.4f}")
        if not mid_period_df.empty:
            mid_vol_ratio = mid_period_df['vol'].sum() / total_volume_safe
            ofi_factor = 0.0
            if level5_df is not None:
                mid_l5 = level5_df[(level5_df.index.time >= time(10, 30)) & (level5_df.index.time < time(14, 30))]
                if not mid_l5.empty and 'ofi' in mid_l5.columns:
                    abs_ofi_per_vol = mid_l5['ofi'].abs().sum() / mid_period_df['vol'].sum() if mid_period_df['vol'].sum() > 0 else 0
                    ofi_factor = np.log1p(abs_ofi_per_vol)
            results['midday_consolidation_level'] = (1 - mid_vol_ratio) * (1 + ofi_factor)
            if enable_probe and is_target_date:
                print(f"--- [探针 ASM.{trade_date_str}] midday_consolidation_level (控盘) ---")
                print(f"    - 原料: 盘中成交量占比={mid_vol_ratio:.4f}, 盘中OFI因子={ofi_factor:.4f}")
                print(f"    - 计算: (1 - {mid_vol_ratio:.4f}) * (1 + {ofi_factor:.4f})")
                print(f"    -> 结果: {results.get('midday_consolidation_level', np.nan):.4f}")
        if not tail_period_df.empty and not mid_period_df.empty and mid_period_df['vol'].mean() > 0:
            accel_ratio = tail_period_df['vol'].mean() / mid_period_df['vol'].mean()
            tail_thrust_purity = np.nan
            if tick_df is not None:
                tail_ticks = tick_df[tick_df.index.time >= time(14, 30)]
                if not tail_ticks.empty and tail_ticks['volume'].sum() > 0:
                    price_diff = tail_ticks['price'].diff().fillna(0)
                    net_thrust_vol = (tail_ticks['volume'] * np.sign(price_diff)).sum()
                    tail_thrust_purity = net_thrust_vol / tail_ticks['volume'].sum()
            if pd.notna(tail_thrust_purity):
                results['tail_volume_acceleration'] = accel_ratio * tail_thrust_purity
            else:
                results['tail_volume_acceleration'] = accel_ratio * np.sign(tail_period_df['close'].iloc[-1] - tail_period_df['open'].iloc[0])
            if enable_probe and is_target_date:
                print(f"--- [探针 ASM.{trade_date_str}] tail_volume_acceleration (控盘) ---")
                print(f"    - 原料: 尾盘/盘中成交均值比={accel_ratio:.2f}, 尾盘推力纯度={tail_thrust_purity:.4f}")
                print(f"    - 计算: {accel_ratio:.2f} * {tail_thrust_purity:.4f}")
                print(f"    -> 结果: {results.get('tail_volume_acceleration', np.nan):.4f}")
            vpoc = context.get('_today_vpoc', day_close_qfq)
            if pd.notna(vpoc):
                deviation_magnitude = (day_close_qfq - vpoc) / atr_14
                tail_force_factor = np.log1p(accel_ratio)
                conviction_purity = tail_thrust_purity if pd.notna(tail_thrust_purity) else np.sign(day_close_qfq - vpoc)
                results['closing_conviction_score'] = deviation_magnitude * tail_force_factor * conviction_purity
                if enable_probe and is_target_date:
                    print(f"--- [探针 ASM.{trade_date_str}] closing_conviction_score (控盘) ---")
                    print(f"    - 原料: 收盘价={day_close_qfq:.2f}, VPOC={vpoc:.2f}, ATR={atr_14:.4f}")
                    print(f"    - 节点: VPOC偏离={deviation_magnitude:.4f}, 尾盘力量因子={tail_force_factor:.4f}, 信念纯度={conviction_purity:.4f}")
                    print(f"    - 计算: {deviation_magnitude:.4f} * {tail_force_factor:.4f} * {conviction_purity:.4f}")
                    print(f"    -> 结果: {results.get('closing_conviction_score', np.nan):.4f}")
        return results

    @staticmethod
    def calculate_game_efficiency_metrics(context: dict) -> dict:
        """
        计算博弈效率相关指标。
        【V42.0 · 动能归一】
        - 核心升级: 将 `distribution_pressure_index` 和 `absorption_strength_index` 的计算内核，
                     从 B/S 盘意图推断，全面升级为基于“动能回溯”的真实动能度量。
                     这使得指标能更精准地量化上涨中的真实派发压力和下跌中的真实吸筹强度。
        """
        group = context['group']
        tick_df = context.get('tick_df')
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        results = {}
        if tick_df is None or tick_df.empty or group.empty:
            return results
        # 全面升级为基于“动能回溯”的计算逻辑
        # 准备分钟级和Tick级数据
        group['price_change_minute'] = group['close'].diff()
        if 'price_change' in tick_df.columns and not tick_df['price_change'].isnull().all():
            self_calculated_change = tick_df['price'].diff().fillna(0)
            zero_change_mask = tick_df['price_change'] == 0
            tick_df['effective_price_change'] = np.where(zero_change_mask, self_calculated_change, tick_df['price_change'])
            tick_df['thrust_direction'] = np.sign(tick_df['effective_price_change'])
            # 1. 上涨派发压力指数 (Distribution Pressure Index)
            up_minutes_index = group[group['price_change_minute'] > 0].index
            up_minutes_ticks = tick_df[tick_df.index.floor('T').isin(up_minutes_index)]
            if not up_minutes_ticks.empty:
                downward_thrust_vol = up_minutes_ticks[up_minutes_ticks['thrust_direction'] < 0]['volume'].sum()
                upward_thrust_vol = up_minutes_ticks[up_minutes_ticks['thrust_direction'] > 0]['volume'].sum()
                if upward_thrust_vol > 0:
                    pressure_index = downward_thrust_vol / upward_thrust_vol
                    results['distribution_pressure_index'] = pressure_index
                    if enable_probe and is_target_date:
                        print(f"--- [探针 ASM.{trade_date_str}] distribution_pressure_index (高频-动能归一) ---")
                        print(f"    - 原料: 上涨分钟内的向下动能成交量(真实派发)={downward_thrust_vol:,.0f}, 向上动能成交量(真实驱动)={upward_thrust_vol:,.0f}")
                        print(f"    - 计算: {downward_thrust_vol:,.0f} / {upward_thrust_vol:,.0f}")
                        print(f"    -> 结果: {pressure_index:.4f}")
            # 2. 下跌吸筹强度指数 (Absorption Strength Index)
            down_minutes_index = group[group['price_change_minute'] < 0].index
            down_minutes_ticks = tick_df[tick_df.index.floor('T').isin(down_minutes_index)]
            if not down_minutes_ticks.empty:
                upward_thrust_vol = down_minutes_ticks[down_minutes_ticks['thrust_direction'] > 0]['volume'].sum()
                downward_thrust_vol = down_minutes_ticks[down_minutes_ticks['thrust_direction'] < 0]['volume'].sum()
                if downward_thrust_vol > 0:
                    strength_index = upward_thrust_vol / downward_thrust_vol
                    results['absorption_strength_index'] = strength_index
                    if enable_probe and is_target_date:
                        print(f"--- [探针 ASM.{trade_date_str}] absorption_strength_index (高频-动能归一) ---")
                        print(f"    - 原料: 下跌分钟内的向上动能成交量(真实抵抗)={upward_thrust_vol:,.0f}, 向下动能成交量(真实驱动)={downward_thrust_vol:,.0f}")
                        print(f"    - 计算: {upward_thrust_vol:,.0f} / {downward_thrust_vol:,.0f}")
                        print(f"    -> 结果: {strength_index:.4f}")
        else: # 回退到旧的B/S盘逻辑
            group['vol_buy'] = tick_df[tick_df['type'] == 'B']['volume'].resample('T').sum()
            group['vol_sell'] = tick_df[tick_df['type'] == 'S']['volume'].resample('T').sum()
            group.fillna(0, inplace=True)
            up_minutes = group[(group['price_change_minute'] > 0) & (group['vol_buy'] > group['vol_sell'])]
            if not up_minutes.empty:
                distribution_vol = up_minutes['vol_sell'].sum()
                driving_vol = up_minutes['vol_buy'].sum()
                if driving_vol > 0:
                    results['distribution_pressure_index'] = distribution_vol / driving_vol
            down_minutes = group[(group['price_change_minute'] < 0) & (group['vol_sell'] > group['vol_buy'])]
            down_minutes_ticks = tick_df[tick_df.index.floor('T').isin(down_minutes.index)]
            if not down_minutes_ticks.empty:
                absorption_vol = down_minutes_ticks[down_minutes_ticks['type'] == 'B']['volume'].sum()
                driving_vol = down_minutes_ticks[down_minutes_ticks['type'] == 'S']['volume'].sum()
                if driving_vol > 0:
                    results['absorption_strength_index'] = absorption_vol / driving_vol
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












