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
        【V30.17 · 索引访问模式修复】
        - 核心升级: 完成对本指标组剩余三个指标(反转动能、高位整固、开盘推力)的高频穿透升级。
        - 核心修复: 修正了在分钟数据回退逻辑中，因idxmin()返回Timestamp而iloc需要整数位置导致的TypeError。
        - 核心修复: 将所有对 'trade_time' 列的访问改为对 DataFrame 索引的访问，解决KeyError。
        """
        group = context['group']
        daily_series_for_day = context['daily_series_for_day']
        atr_14 = context['atr_14']
        tick_df = context.get('tick_df')
        level5_df = context.get('level5_df')
        day_open_qfq = context['day_open_qfq']
        day_high_qfq = context['day_high_qfq']
        day_low_qfq = context['day_low_qfq']
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
            'rebound_momentum': np.nan,
            'high_level_consolidation_volume': np.nan,
            'opening_period_thrust': np.nan,
        }
        if pd.notna(atr_14) and atr_14 > 0:
            turnover_rate = pd.to_numeric(daily_series_for_day.get('turnover_rate_f'), errors='coerce')
            if pd.notna(turnover_rate):
                results['intraday_energy_density'] = np.log1p(turnover_rate) / atr_14
        if tick_df is not None and not tick_df.empty and 'type' in tick_df.columns:
            total_volume = tick_df['volume'].sum()
            if total_volume > 0:
                active_buy_vol = tick_df[tick_df['type'] == 'B']['volume'].sum()
                active_sell_vol = tick_df[tick_df['type'] == 'S']['volume'].sum()
                results['intraday_thrust_purity'] = (active_buy_vol - active_sell_vol) / total_volume
                if enable_probe and is_target_date:
                    print(f"--- [探针 ASM.{trade_date_str}] intraday_thrust_purity (高频) ---")
                    print(f"    - 原料: 总成交量={total_volume:,.0f}, 主动买量={active_buy_vol:,.0f}, 主动卖量={active_sell_vol:,.0f}")
                    print(f"    - 计算: ({active_buy_vol:,.0f} - {active_sell_vol:,.0f}) / {total_volume:,.0f}")
                    print(f"    -> 结果: {results['intraday_thrust_purity']:.4f}")
        else:
            thrust_vector = (group['close'] - group['open']) * group['vol']
            absolute_energy = abs(group['close'] - group['open']) * group['vol']
            total_energy = absolute_energy.sum()
            if total_energy > 0:
                results['intraday_thrust_purity'] = thrust_vector.sum() / total_energy
                if enable_probe and is_target_date:
                    print(f"--- [探针 ASM.{trade_date_str}] intraday_thrust_purity (分钟降级) ---")
                    print(f"    - 原料: 推力向量和={thrust_vector.sum():,.2f}, 绝对能量和={total_energy:,.2f}")
                    print(f"    - 计算: {thrust_vector.sum():,.2f} / {total_energy:,.2f}")
                    print(f"    -> 结果: {results['intraday_thrust_purity']:.4f}")
        if tick_df is not None and not tick_df.empty:
            results['volume_burstiness_index'] = StructuralMetricsCalculators.calculate_gini(tick_df['volume'].values)
            if enable_probe and is_target_date:
                print(f"--- [探针 ASM.{trade_date_str}] volume_burstiness_index (高频) ---")
                print(f"    - 原料: {len(tick_df)} 笔逐笔成交量序列")
                print(f"    -> 结果: {results['volume_burstiness_index']:.4f}")
        else:
            results['volume_burstiness_index'] = StructuralMetricsCalculators.calculate_gini(group['vol'].values)
            if enable_probe and is_target_date:
                print(f"--- [探针 ASM.{trade_date_str}] volume_burstiness_index (分钟降级) ---")
                print(f"    - 原料: {len(group)} 根分钟线成交量序列")
                print(f"    -> 结果: {results['volume_burstiness_index']:.4f}")
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
                        results['auction_impact_score'] = gap_magnitude * (1 + conviction_factor)
                        if enable_probe and is_target_date:
                            print(f"--- [探针 ASM.{trade_date_str}] auction_impact_score (高频) ---")
                            print(f"    - 原料: 开盘价={day_open_qfq:.2f}, 昨收={pre_close_qfq:.2f}, ATR={atr_14:.4f}")
                            print(f"    - 节点1 (缺口): ({day_open_qfq:.2f} - {pre_close_qfq:.2f}) / {atr_14:.4f} = {gap_magnitude:.4f}")
                            print(f"    - 原料2: 开盘5分钟OFI={opening_ofi:,.0f}, 成交量={opening_volume:,.0f}")
                            print(f"    - 节点2 (信念): tanh({opening_ofi:,.0f} / {opening_volume:,.0f}) = {conviction_factor:.4f}")
                            print(f"    - 计算: {gap_magnitude:.4f} * (1 + {conviction_factor:.4f})")
                            print(f"    -> 结果: {results['auction_impact_score']:.4f}")
                    else:
                        results['auction_impact_score'] = gap_magnitude
                else:
                    results['auction_impact_score'] = gap_magnitude
            else:
                results['auction_impact_score'] = gap_magnitude
                if enable_probe and is_target_date:
                    print(f"--- [探针 ASM.{trade_date_str}] auction_impact_score (分钟降级) ---")
                    print(f"    - 原料: 开盘价={day_open_qfq:.2f}, 昨收={pre_close_qfq:.2f}, ATR={atr_14:.4f}")
                    print(f"    - 计算: ({day_open_qfq:.2f} - {pre_close_qfq:.2f}) / {atr_14:.4f}")
                    print(f"    -> 结果: {results['auction_impact_score']:.4f}")
        # 新增代码块：rebound_momentum (反转动能)
        if tick_df is not None and not tick_df.empty:
            low_price_time = tick_df['price'].idxmin()
            falling_ticks = tick_df.loc[:low_price_time]
            rebounding_ticks = tick_df.loc[low_price_time:]
            if not falling_ticks.empty and not rebounding_ticks.empty and falling_ticks['volume'].sum() > 0 and rebounding_ticks['volume'].sum() > 0:
                vwap_fall = (falling_ticks['price'] * falling_ticks['volume']).sum() / falling_ticks['volume'].sum()
                vwap_rebound = (rebounding_ticks['price'] * rebounding_ticks['volume']).sum() / rebounding_ticks['volume'].sum()
                rebounding_vol_ratio = rebounding_ticks['volume'].sum() / tick_df['volume'].sum()
                if vwap_fall > 0:
                    results['rebound_momentum'] = (vwap_rebound / vwap_fall - 1) * rebounding_vol_ratio * 100
                    if enable_probe and is_target_date:
                        print(f"--- [探针 ASM.{trade_date_str}] rebound_momentum (高频) ---")
                        print(f"    - 原料: 转折点={low_price_time.time()}, 下跌VWAP={vwap_fall:.4f}, 反弹VWAP={vwap_rebound:.4f}, 反弹量占比={rebounding_vol_ratio:.2%}")
                        print(f"    - 计算: ({vwap_rebound:.4f} / {vwap_fall:.4f} - 1) * {rebounding_vol_ratio:.4f} * 100")
                        print(f"    -> 结果: {results['rebound_momentum']:.4f}")
        else:
            low_timestamp = group['low'].idxmin()
            low_pos = group.index.get_loc(low_timestamp)
            if low_pos > 0 and low_pos < len(group) - 1:
                falling_phase = group.iloc[:low_pos+1]
                rebounding_phase = group.iloc[low_pos+1:]
                vwap_fall = (falling_phase['amount']).sum() / falling_phase['vol'].sum() if falling_phase['vol'].sum() > 0 else np.nan
                vwap_rebound = (rebounding_phase['amount']).sum() / rebounding_phase['vol'].sum() if rebounding_phase['vol'].sum() > 0 else np.nan
                if pd.notna(vwap_fall) and pd.notna(vwap_rebound) and vwap_fall > 0 and group['vol'].sum() > 0:
                    results['rebound_momentum'] = (vwap_rebound / vwap_fall - 1) * (rebounding_phase['vol'].sum() / group['vol'].sum()) * 100
                    if enable_probe and is_target_date:
                        print(f"--- [探针 ASM.{trade_date_str}] rebound_momentum (分钟降级) ---")
                        print(f"    -> 结果: {results['rebound_momentum']:.4f}")
        # 新增代码块：high_level_consolidation_volume (高位整固成交量占比)
        price_range = day_high_qfq - day_low_qfq
        if price_range > 0:
            high_level_threshold = day_high_qfq - 0.25 * price_range
            if tick_df is not None and not tick_df.empty:
                high_vol = tick_df[tick_df['price'] >= high_level_threshold]['volume'].sum()
                total_vol = tick_df['volume'].sum()
                if total_vol > 0:
                    results['high_level_consolidation_volume'] = high_vol / total_vol
                    if enable_probe and is_target_date:
                        print(f"--- [探针 ASM.{trade_date_str}] high_level_consolidation_volume (高频) ---")
                        print(f"    - 原料: 高位阈值={high_level_threshold:.2f}, 高位成交量={high_vol:,.0f}, 总量={total_vol:,.0f}")
                        print(f"    -> 结果: {results['high_level_consolidation_volume']:.4f}")
            else:
                total_volume = group['vol'].sum()
                if total_volume > 0:
                    results['high_level_consolidation_volume'] = group[group['high'] >= high_level_threshold]['vol'].sum() / total_volume
                    if enable_probe and is_target_date:
                        print(f"--- [探针 ASM.{trade_date_str}] high_level_consolidation_volume (分钟降级) ---")
                        print(f"    -> 结果: {results['high_level_consolidation_volume']:.4f}")
        # 新增代码块：opening_period_thrust (开盘期推力)
        if tick_df is not None and not tick_df.empty and 'type' in tick_df.columns:
            opening_ticks = tick_df.between_time('09:30:00', '09:59:59')
            if not opening_ticks.empty:
                opening_total_vol = opening_ticks['volume'].sum()
                if opening_total_vol > 0:
                    opening_buy_vol = opening_ticks[opening_ticks['type'] == 'B']['volume'].sum()
                    opening_sell_vol = opening_ticks[opening_ticks['type'] == 'S']['volume'].sum()
                    results['opening_period_thrust'] = (opening_buy_vol - opening_sell_vol) / opening_total_vol
                    if enable_probe and is_target_date:
                        print(f"--- [探针 ASM.{trade_date_str}] opening_period_thrust (高频) ---")
                        print(f"    - 原料: 开盘买量={opening_buy_vol:,.0f}, 开盘卖量={opening_sell_vol:,.0f}, 开盘总量={opening_total_vol:,.0f}")
                        print(f"    - 计算: ({opening_buy_vol:,.0f} - {opening_sell_vol:,.0f}) / {opening_total_vol:,.0f}")
                        print(f"    -> 结果: {results['opening_period_thrust']:.4f}")
        else:
            # 修改代码行：将 group['trade_time'].dt.time 替换为 group.index.time
            opening_period_df = group[group.index.time < time(9, 59, 59)]
            if not opening_period_df.empty:
                opening_thrust_vector = (opening_period_df['close'] - opening_period_df['open']) * opening_period_df['vol']
                opening_absolute_energy = abs(opening_period_df['close'] - opening_period_df['open']) * opening_period_df['vol']
                if opening_absolute_energy.sum() > 0:
                    results['opening_period_thrust'] = opening_thrust_vector.sum() / opening_absolute_energy.sum()
                    if enable_probe and is_target_date:
                        print(f"--- [探针 ASM.{trade_date_str}] opening_period_thrust (分钟降级) ---")
                        print(f"    -> 结果: {results['opening_period_thrust']:.4f}")
        return results

    @staticmethod
    def calculate_control_metrics(context: dict) -> dict:
        """
        【V36.2 · 完整性调用修正】
        - 核心重构: 移除了 `intraday_thrust_purity` 的冗余计算，该指标的计算职责已明确归于 `calculate_energy_density_metrics`。
        """
        group = context['group']
        tick_df = context.get('tick_df')
        day_close_qfq = context['day_close_qfq']
        total_volume_safe = context['total_volume_safe']
        atr_14 = context['atr_14']
        results = {}
        if total_volume_safe == 0:
            return results
        # 1. 趋势效率 (Trend Efficiency Ratio) - (原第3点，逻辑上移)
        if not group.empty:
            net_displacement = abs(day_close_qfq - group['open'].iloc[0])
            total_path = group['high'].max() - group['low'].min()
            if total_path > 0:
                results['trend_efficiency_ratio'] = net_displacement / total_path
        # 2. 均值回归频率 (Mean Reversion Frequency)
        if not group.empty:
            vwap = (group['amount'] * 10000).cumsum() / group['vol'].cumsum()
            price_cross_vwap = np.sum(np.diff(np.sign(group['close'] - vwap)) != 0)
            trading_hours = (group.index[-1] - group.index[0]).total_seconds() / 3600
            if trading_hours > 0:
                results['mean_reversion_frequency'] = price_cross_vwap / trading_hours
        # 3. 日内盈亏失衡度 (Intraday PNL Imbalance)
        if tick_df is not None and not tick_df.empty:
            winning_vol = tick_df[tick_df['price'] < day_close_qfq]['volume'].sum()
            losing_vol = tick_df[tick_df['price'] > day_close_qfq]['volume'].sum()
            traded_vol = winning_vol + losing_vol
            if traded_vol > 0:
                results['intraday_pnl_imbalance'] = (winning_vol - losing_vol) / traded_vol
        # 4. 成本离散度指数 (Cost Dispersion Index)
        if not group.empty and pd.notna(atr_14) and atr_14 > 0:
            day_vwap = (group['amount'].sum() * 10000) / group['vol'].sum()
            weighted_variance = ((group['close'] - day_vwap) ** 2 * group['vol']).sum() / group['vol'].sum()
            weighted_std_dev = np.sqrt(weighted_variance)
            results['cost_dispersion_index'] = weighted_std_dev / atr_14
        return results

    @staticmethod
    def calculate_game_efficiency_metrics(context: dict) -> dict:
        """
        【V29.0 · 节律与偏度穿透】
        - 核心升级: 为 `volatility_skew_index` 指标实现高频穿透，基于tick收益率精确度量日内波动情绪的偏向。
        """
        group = context['group']
        continuous_group = context['continuous_group']
        tick_df = context.get('tick_df')
        day_open_qfq = context['day_open_qfq']
        total_volume_safe = context['total_volume_safe']
        atr_14 = context['atr_14']
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        results = {}
        if tick_df is not None and not tick_df.empty and 'type' in tick_df.columns:
            tick_df['price_diff'] = tick_df['price'].diff().fillna(0)
            active_buys = tick_df[tick_df['type'] == 'B']
            total_active_buy_vol = active_buys['volume'].sum()
            if total_active_buy_vol > 0:
                price_gain_by_buys = active_buys[active_buys['price_diff'] > 0]['price_diff'].sum()
                results['upward_thrust_efficacy'] = (price_gain_by_buys / total_active_buy_vol) * 10000
                if enable_probe and is_target_date:
                    print(f"--- [探针 ASM.{trade_date_str}] upward_thrust_efficacy (高频) ---")
                    print(f"    - 原料: 主动买盘驱动的价格上涨总和={price_gain_by_buys:.4f}, 主动买盘总成交量={total_active_buy_vol:,.0f}")
                    print(f"    - 计算: ({price_gain_by_buys:.4f} / {total_active_buy_vol:,.0f}) * 10000")
                    print(f"    -> 结果: {results['upward_thrust_efficacy']:.4f}")
            price_is_falling_mask = tick_df['price_diff'] < 0
            passive_sells_in_fall = tick_df[price_is_falling_mask & (tick_df['type'] == 'S')]
            active_buys_in_fall = tick_df[price_is_falling_mask & (tick_df['type'] == 'B')]
            total_passive_sell_vol_in_fall = passive_sells_in_fall['volume'].sum()
            if total_passive_sell_vol_in_fall > 0:
                total_active_buy_vol_in_fall = active_buys_in_fall['volume'].sum()
                results['downward_absorption_efficacy'] = total_active_buy_vol_in_fall / total_passive_sell_vol_in_fall
                if enable_probe and is_target_date:
                    print(f"--- [探针 ASM.{trade_date_str}] downward_absorption_efficacy (高频) ---")
                    print(f"    - 原料: 下跌中的主动买量(抵抗)={total_active_buy_vol_in_fall:,.0f}, 下跌中的被动卖量(驱动)={total_passive_sell_vol_in_fall:,.0f}")
                    print(f"    - 计算: {total_active_buy_vol_in_fall:,.0f} / {total_passive_sell_vol_in_fall:,.0f}")
                    print(f"    -> 结果: {results['downward_absorption_efficacy']:.4f}")
        else:
            continuous_group['price_diff'] = continuous_group['close'] - continuous_group['open']
            up_minutes = continuous_group[continuous_group['price_diff'] > 0]
            if not up_minutes.empty and up_minutes['vol'].sum() > 0 and pd.notna(total_volume_safe) and day_open_qfq > 0:
                normalized_price_gain = up_minutes['price_diff'].sum() / day_open_qfq
                normalized_volume_cost = up_minutes['vol'].sum() / total_volume_safe
                if normalized_volume_cost > 0: results['upward_thrust_efficacy'] = normalized_price_gain / normalized_volume_cost
            down_minutes = continuous_group[continuous_group['price_diff'] < 0]
            if not down_minutes.empty and abs(down_minutes['price_diff']).sum() > 0 and pd.notna(total_volume_safe) and day_open_qfq > 0:
                normalized_price_drop = abs(down_minutes['price_diff']).sum() / day_open_qfq
                normalized_volume_cost = down_minutes['vol'].sum() / total_volume_safe
                if normalized_price_drop > 0: results['downward_absorption_efficacy'] = normalized_volume_cost / normalized_price_drop
            if enable_probe and is_target_date:
                print(f"--- [探针 ASM.{trade_date_str}] 博弈效能 (分钟降级) ---")
                print(f"    -> upward_thrust_efficacy: {results.get('upward_thrust_efficacy', np.nan):.4f}")
                print(f"    -> downward_absorption_efficacy: {results.get('downward_absorption_efficacy', np.nan):.4f}")
        up_eff = results.get('upward_thrust_efficacy')
        down_eff = results.get('downward_absorption_efficacy')
        if all(pd.notna(v) for v in [up_eff, down_eff]) and up_eff > 0 and down_eff > 0:
            results['net_vpa_score'] = np.log(up_eff / down_eff)
        if len(continuous_group) >= 30 and pd.notna(atr_14) and atr_14 > 0:
            from scipy.signal import find_peaks
            from itertools import combinations
            price_series = continuous_group['minute_vwap']
            rsi_series = ta.rsi(price_series, length=14).dropna()
            if not rsi_series.empty:
                aligned_price, aligned_rsi = price_series.loc[rsi_series.index], rsi_series
                price_low_indices, _ = find_peaks(-aligned_price.values, distance=15, prominence=aligned_price.std()*0.5)
                rsi_low_indices, _ = find_peaks(-aligned_rsi.values, distance=15, prominence=aligned_rsi.std()*0.5)
                price_high_indices, _ = find_peaks(aligned_price.values, distance=15, prominence=aligned_price.std()*0.5)
                rsi_high_indices, _ = find_peaks(aligned_rsi.values, distance=15, prominence=aligned_rsi.std()*0.5)
                bullish_strengths, bearish_strengths = [], []
                if len(price_low_indices) >= 2 and len(rsi_low_indices) >= 2:
                    for i1, i2 in combinations(price_low_indices, 2):
                        if aligned_price.iloc[i2] < aligned_price.iloc[i1]:
                            try:
                                r1 = aligned_rsi.iloc[rsi_low_indices[np.abs(rsi_low_indices - i1).argmin()]]
                                r2 = aligned_rsi.iloc[rsi_low_indices[np.abs(rsi_low_indices - i2).argmin()]]
                                if r2 > r1: bullish_strengths.append(((aligned_price.iloc[i1] - aligned_price.iloc[i2]) / atr_14) * (r2 - r1))
                            except IndexError: continue
                if len(price_high_indices) >= 2 and len(rsi_high_indices) >= 2:
                    for i1, i2 in combinations(price_high_indices, 2):
                        if aligned_price.iloc[i2] > aligned_price.iloc[i1]:
                            try:
                                r1 = aligned_rsi.iloc[rsi_high_indices[np.abs(rsi_high_indices - i1).argmin()]]
                                r2 = aligned_rsi.iloc[rsi_high_indices[np.abs(rsi_high_indices - i2).argmin()]]
                                if r2 < r1: bearish_strengths.append(((aligned_price.iloc[i2] - aligned_price.iloc[i1]) / atr_14) * (r2 - r1))
                            except IndexError: continue
                if bullish_strengths: results['divergence_conviction_score'] = max(bullish_strengths)
                elif bearish_strengths: results['divergence_conviction_score'] = min(bearish_strengths)
        # 为 volatility_skew_index 增加高频计算逻辑
        if tick_df is not None and not tick_df.empty:
            tick_returns = tick_df['price'].pct_change().fillna(0)
            weights = tick_df['volume']
            if weights.sum() > 0:
                weighted_mean = np.average(tick_returns, weights=weights)
                weighted_var = np.average((tick_returns - weighted_mean)**2, weights=weights)
                if weighted_var > 0:
                    weighted_std = np.sqrt(weighted_var)
                    weighted_skew = np.average(((tick_returns - weighted_mean) / weighted_std)**3, weights=weights)
                    results['volatility_skew_index'] = weighted_skew
                    if enable_probe and is_target_date:
                        print(f"--- [探针 ASM.{trade_date_str}] volatility_skew_index (高频) ---")
                        print(f"    - 原料: {len(tick_returns)} 笔高频收益率, 总成交量={weights.sum():,.0f}")
                        print(f"    - 节点: 加权均值={weighted_mean:.6f}, 加权标准差={weighted_std:.6f}")
                        print(f"    -> 结果: {results['volatility_skew_index']:.4f}")
        else:
            returns = continuous_group['minute_vwap'].pct_change().fillna(0)
            weights = continuous_group['vol']
            if weights.sum() > 0:
                weighted_mean = np.average(returns, weights=weights)
                weighted_var = np.average((returns - weighted_mean)**2, weights=weights)
                if weighted_var > 0:
                    weighted_std = np.sqrt(weighted_var)
                    results['volatility_skew_index'] = np.average(((returns - weighted_mean) / weighted_std)**3, weights=weights)
                    if enable_probe and is_target_date:
                        print(f"--- [探针 ASM.{trade_date_str}] volatility_skew_index (分钟降级) ---")
                        print(f"    -> 结果: {results['volatility_skew_index']:.4f}")
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












