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
        【V37.1 · 动能纯化修正】
        - 核心修复: 移除了 `rebound_momentum` 计算中引入严重时间偏误的“反弹量占比”因子。
                     修正后的指标 `(vwap_rebound / vwap_fall - 1) * 100` 更纯粹地衡量了
                     从日内低点反弹的真实动能，消除了低点出现时间对结果的干扰。
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
        # rebound_momentum (反转动能)
        if tick_df is not None and not tick_df.empty:
            low_price_time = tick_df['price'].idxmin()
            falling_ticks = tick_df.loc[:low_price_time]
            rebounding_ticks = tick_df.loc[low_price_time:]
            if not falling_ticks.empty and not rebounding_ticks.empty and falling_ticks['volume'].sum() > 0 and rebounding_ticks['volume'].sum() > 0:
                vwap_fall = (falling_ticks['price'] * falling_ticks['volume']).sum() / falling_ticks['volume'].sum()
                vwap_rebound = (rebounding_ticks['price'] * rebounding_ticks['volume']).sum() / rebounding_ticks['volume'].sum()
                if vwap_fall > 0:
                    # 修改代码行：移除引入时间偏误的 rebounding_vol_ratio 因子
                    results['rebound_momentum'] = (vwap_rebound / vwap_fall - 1) * 100
                    if enable_probe and is_target_date:
                        print(f"--- [探针 ASM.{trade_date_str}] rebound_momentum (高频) ---")
                        # 修改代码行：更新探针日志，不再显示“反弹量占比”
                        print(f"    - 原料: 转折点={low_price_time.time()}, 下跌VWAP={vwap_fall:.4f}, 反弹VWAP={vwap_rebound:.4f}")
                        # 修改代码行：更新计算过程的日志说明
                        print(f"    - 计算: ({vwap_rebound:.4f} / {vwap_fall:.4f} - 1) * 100")
                        print(f"    -> 结果: {results['rebound_momentum']:.4f}")
        else:
            low_timestamp = group['low'].idxmin()
            low_pos = group.index.get_loc(low_timestamp)
            if low_pos > 0 and low_pos < len(group) - 1:
                falling_phase = group.iloc[:low_pos+1]
                rebounding_phase = group.iloc[low_pos+1:]
                vwap_fall = (falling_phase['amount']).sum() / falling_phase['vol'].sum() if falling_phase['vol'].sum() > 0 else np.nan
                vwap_rebound = (rebounding_phase['amount']).sum() / rebounding_phase['vol'].sum() if rebounding_phase['vol'].sum() > 0 else np.nan
                if pd.notna(vwap_fall) and pd.notna(vwap_rebound) and vwap_fall > 0:
                    # 修改代码行：移除引入时间偏误的成交量占比因子
                    results['rebound_momentum'] = (vwap_rebound / vwap_fall - 1) * 100
                    if enable_probe and is_target_date:
                        print(f"--- [探针 ASM.{trade_date_str}] rebound_momentum (分钟降级) ---")
                        print(f"    -> 结果: {results['rebound_momentum']:.4f}")
        # high_level_consolidation_volume (高位整固成交量占比)
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
        # opening_period_thrust (开盘期推力)
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
        计算博弈效率相关指标。
        【V36.9 · 对称归因修正】
        - 核心修复: 对 `downward_absorption_efficacy` 应用与上行指标对称的归因逻辑。
                     现在只统计价格下跌且主动卖盘力量大于主动买盘力量的分钟，
                     确保指标衡量的是在空方主导的真实下行压力中的吸收强度。
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
        # 1. 上行冲击效率 (Upward Thrust Efficacy) - V36.8 已修正
        up_minutes = group[(group['price_change'] > 0) & (group['vol_buy'] > group['vol_sell'])]
        if not up_minutes.empty:
            total_price_increase = up_minutes['price_change'].sum()
            total_buy_vol_in_up_minutes = up_minutes['vol_buy'].sum()
            if total_buy_vol_in_up_minutes > 0:
                efficacy = (total_price_increase / total_buy_vol_in_up_minutes) * 10000
                results['upward_thrust_efficacy'] = efficacy
                if enable_probe and is_target_date:
                    print(f"--- [探针 ASM.{trade_date_str}] upward_thrust_efficacy (高频) ---")
                    print(f"    - 原料: 主动买盘驱动的价格上涨总和={total_price_increase:.4f}, 主动买盘总成交量={total_buy_vol_in_up_minutes:,.0f}")
                    print(f"    - 计算: ({total_price_increase:.4f} / {total_buy_vol_in_up_minutes:,.0f}) * 10000")
                    print(f"    -> 结果: {efficacy:.4f}")
        # 2. 下行吸收效率 (Downward Absorption Efficacy)
        # 修改代码行：增加 vol_sell > vol_buy 的筛选条件，进行对称的精确归因
        down_minutes = group[(group['price_change'] < 0) & (group['vol_sell'] > group['vol_buy'])]
        down_minutes_ticks = tick_df[tick_df.index.floor('T').isin(down_minutes.index)]
        if not down_minutes_ticks.empty:
            absorption_vol = down_minutes_ticks[down_minutes_ticks['type'] == 'B']['volume'].sum()
            driving_vol = down_minutes_ticks[down_minutes_ticks['type'] == 'S']['volume'].sum()
            if driving_vol > 0:
                absorption_ratio = absorption_vol / driving_vol
                results['downward_absorption_efficacy'] = absorption_ratio
                if enable_probe and is_target_date:
                    print(f"--- [探针 ASM.{trade_date_str}] downward_absorption_efficacy (高频) ---")
                    print(f"    - 原料: 下跌中的主动买量(抵抗)={absorption_vol:,.0f}, 下跌中的主动卖量(驱动)={driving_vol:,.0f}")
                    print(f"    - 计算: {absorption_vol:,.0f} / {driving_vol:,.0f}")
                    print(f"    -> 结果: {absorption_ratio:.4f}")
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












