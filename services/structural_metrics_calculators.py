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
        【V38.1 · 反转量价效率】
        - 核心重构: 升级 `dynamic_reversal_strength` 的核心动能公式，引入“量价效率”概念。
                     新的动能值 = (价格VWAP增益) * (下跌成交量 / 反弹成交量)。
                     此举旨在嘉奖“缩量反弹”（高效）并惩罚“放量反弹”（低效），
                     使指标从单纯衡量价格结果，进化为评估反转的综合质量与效率。
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
                reversal_momentums = []
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
                                # 修改代码块：引入量价效率计算
                                vol_fall = falling_phase['vol'].sum()
                                vol_rebound = rebounding_phase['vol'].sum()
                                if not falling_phase.empty and not rebounding_phase.empty and \
                                   vol_fall > 0 and vol_rebound > 0:
                                    vwap_fall = falling_phase['amount'].sum() / vol_fall
                                    vwap_rebound = rebounding_phase['amount'].sum() / vol_rebound
                                    if vwap_fall > 0:
                                        price_momentum = (vwap_rebound / vwap_fall - 1)
                                        volume_efficiency = vol_fall / vol_rebound
                                        momentum = (price_momentum * volume_efficiency) * 100
                                        reversal_momentums.append(momentum)
                if reversal_momentums:
                    successful_reversals = [m for m in reversal_momentums if m > 0]
                    total_attempts = len(reversal_momentums)
                    successful_attempts = len(successful_reversals)
                    if total_attempts > 0:
                        conviction_rate = successful_attempts / total_attempts
                        results['reversal_conviction_rate'] = conviction_rate
                        if enable_probe and is_target_date:
                            print(f"--- [探针 ASM.{trade_date_str}] reversal_conviction_rate (分钟) ---")
                            print(f"    - 前置: 使用 {prominence_source} 显著性阈值 = {dynamic_prominence:.4f}")
                            print(f"    - 原料: 成功次数={successful_attempts}, 总尝试次数={total_attempts}")
                            print(f"    - 计算: {successful_attempts} / {total_attempts}")
                            print(f"    -> 结果: {conviction_rate:.4f}")
                    if successful_reversals:
                        results['dynamic_reversal_strength'] = np.mean(successful_reversals)
                        if enable_probe and is_target_date:
                            print(f"--- [探针 ASM.{trade_date_str}] dynamic_reversal_strength (分钟) ---")
                            print(f"    - 前置: 使用 {prominence_source} 显著性阈值 = {dynamic_prominence:.4f}")
                            print(f"    - 原料: 识别出 {total_attempts} 次反转尝试, 其中 {successful_attempts} 次成功")
                            # 修改代码行：更新探针日志以反映新的量价效率动能
                            print(f"    - 节点 (量价效率动能): {[f'{m:.2f}' for m in successful_reversals]}")
                            print(f"    - 计算: mean of successful reversals")
                            print(f"    -> 结果: {results['dynamic_reversal_strength']:.4f}")
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
        计算博弈效率相关指标。
        【V37.9 · 博弈内核归一】
        - 核心重构: 将此方法确立为 `distribution_pressure_index` 和 `absorption_strength_index` 的唯一计算源。
                     废弃了在其他模块中的冗余、宽泛计算，并正式将当前严谨的计算逻辑（基于价量同向分钟）
                     的结果赋予最终指标名，实现全系统在博弈效率评估上的逻辑统一和架构纯粹。
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
        # 准备分钟级数据
        group['price_change'] = group['close'].diff()
        group['vol_buy'] = tick_df[tick_df['type'] == 'B']['volume'].resample('T').sum()
        group['vol_sell'] = tick_df[tick_df['type'] == 'S']['volume'].resample('T').sum()
        group.fillna(0, inplace=True)
        # 1. 上涨派发压力指数 (Distribution Pressure Index) - V37.9 内核归一
        up_minutes = group[(group['price_change'] > 0) & (group['vol_buy'] > group['vol_sell'])]
        if not up_minutes.empty:
            distribution_vol = up_minutes['vol_sell'].sum()
            driving_vol = up_minutes['vol_buy'].sum()
            if driving_vol > 0:
                pressure_index = distribution_vol / driving_vol
                # 修改代码行：将结果赋予统一后的正式指标名
                results['distribution_pressure_index'] = pressure_index
                if enable_probe and is_target_date:
                    # 修改代码行：更新探针日志的指标名称
                    print(f"--- [探针 ASM.{trade_date_str}] distribution_pressure_index (高频) ---")
                    print(f"    - 原料: 上涨中的主动卖量(派发)={distribution_vol:,.0f}, 上涨中的主动买量(驱动)={driving_vol:,.0f}")
                    print(f"    - 计算: {distribution_vol:,.0f} / {driving_vol:,.0f}")
                    print(f"    -> 结果: {pressure_index:.4f}")
        # 2. 下跌吸筹强度指数 (Absorption Strength Index) - V37.9 内核归一
        down_minutes = group[(group['price_change'] < 0) & (group['vol_sell'] > group['vol_buy'])]
        down_minutes_ticks = tick_df[tick_df.index.floor('T').isin(down_minutes.index)]
        if not down_minutes_ticks.empty:
            absorption_vol = down_minutes_ticks[down_minutes_ticks['type'] == 'B']['volume'].sum()
            driving_vol = down_minutes_ticks[down_minutes_ticks['type'] == 'S']['volume'].sum()
            if driving_vol > 0:
                strength_index = absorption_vol / driving_vol
                # 修改代码行：将结果赋予统一后的正式指标名
                results['absorption_strength_index'] = strength_index
                if enable_probe and is_target_date:
                    # 修改代码行：更新探针日志的指标名称
                    print(f"--- [探针 ASM.{trade_date_str}] absorption_strength_index (高频) ---")
                    print(f"    - 原料: 下跌中的主动买量(抵抗)={absorption_vol:,.0f}, 下跌中的主动卖量(驱动)={driving_vol:,.0f}")
                    print(f"    - 计算: {absorption_vol:,.0f} / {driving_vol:,.0f}")
                    print(f"    -> 结果: {strength_index:.4f}")
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












