# services/microstructure_dynamics_calculators.py
import pandas as pd
import numpy as np
from scipy.stats import norm, linregress

class MicrostructureDynamicsCalculators:
    """
    【V27.0 · 微观动力学内核】
    - 核心职责: 封装所有基于高频数据的微观结构指标计算逻辑，实现最终的内核分离。
    - 架构模式: 作为一个无状态的静态工具类，提供一系列独立的、可诊断的计算函数。
    """
    @staticmethod
    def calculate_all(context: dict) -> dict:
        """主入口：编排所有微观动力学指标的计算"""
        results = {}
        results.update(MicrostructureDynamicsCalculators._calculate_ofi_and_sweeps(context))
        results.update(MicrostructureDynamicsCalculators._calculate_vpin(context))
        results.update(MicrostructureDynamicsCalculators._calculate_hf_mechanics(context))
        results.update(MicrostructureDynamicsCalculators._calculate_liquidity_metrics(context))
        results.update(MicrostructureDynamicsCalculators._calculate_vwap_reversion(context))
        return results

    @staticmethod
    def _calculate_ofi_and_sweeps(context: dict) -> dict:
        """计算订单流失衡(OFI)与扫单强度"""
        tick_df = context.get('tick_df')
        level5_df = context.get('level5_df')
        total_volume = context.get('total_volume_safe')
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        results = {
            'order_flow_imbalance_score': np.nan,
            'buy_sweep_intensity': np.nan,
            'sell_sweep_intensity': np.nan,
        }
        if tick_df is None or tick_df.empty or total_volume == 0:
            return results
        # 订单流失衡 (OFI)
        if level5_df is not None and not level5_df.empty and len(level5_df) > 1:
            df = level5_df[['buy_price1', 'buy_volume1', 'sell_price1', 'sell_volume1']].copy()
            df_prev = df.shift(1)
            delta_buy_price = df['buy_price1'] - df_prev['buy_price1']
            delta_sell_price = df['sell_price1'] - df_prev['sell_price1']
            ofi_static = np.where((delta_buy_price == 0) & (delta_sell_price == 0), df['buy_volume1'] - df_prev['buy_volume1'], 0)
            ofi_dynamic = np.where(delta_buy_price > 0, df_prev['buy_volume1'], 0)
            ofi_dynamic = np.where(delta_buy_price < 0, -df['buy_volume1'], ofi_dynamic)
            ofi_dynamic = np.where(delta_sell_price > 0, ofi_dynamic + df['sell_volume1'], ofi_dynamic)
            ofi_dynamic = np.where(delta_sell_price < 0, ofi_dynamic - df_prev['sell_volume1'], ofi_dynamic)
            ofi_series = ofi_static + ofi_dynamic
            total_ofi = np.nansum(ofi_series)
            if total_volume > 0:
                results['order_flow_imbalance_score'] = total_ofi / total_volume
                if enable_probe and is_target_date:
                    print(f"--- [探针 ASM.{trade_date_str}] order_flow_imbalance_score ---")
                    print(f"    - 原料: 累计OFI={total_ofi:,.0f}, 总成交量={total_volume:,.0f}")
                    print(f"    -> 结果: {results['order_flow_imbalance_score']:.4f}")
        # 扫单强度 (Sweep Intensity)
        buy_sweep_vol, sell_sweep_vol = 0, 0
        min_sweep_len = 3
        tick_df['block'] = (tick_df['type'] != tick_df['type'].shift()).cumsum()
        tick_df['block_size'] = tick_df.groupby('block')['type'].transform('size')
        sweep_candidates = tick_df[(tick_df['block_size'] >= min_sweep_len) & (tick_df['type'].isin(['B', 'S']))]
        if not sweep_candidates.empty:
            for _, group_sweep in sweep_candidates.groupby('block'):
                trade_type = group_sweep['type'].iloc[0]
                prices = group_sweep['price']
                if trade_type == 'B' and prices.is_monotonic_increasing:
                    buy_sweep_vol += group_sweep['volume'].sum()
                elif trade_type == 'S' and prices.is_monotonic_decreasing:
                    sell_sweep_vol += group_sweep['volume'].sum()
        total_buy_vol = tick_df[tick_df['type'] == 'B']['volume'].sum()
        total_sell_vol = tick_df[tick_df['type'] == 'S']['volume'].sum()
        if total_buy_vol > 0:
            results['buy_sweep_intensity'] = buy_sweep_vol / total_buy_vol
            if enable_probe and is_target_date:
                print(f"--- [探针 ASM.{trade_date_str}] buy_sweep_intensity ---")
                print(f"    - 原料: 买方扫单量={buy_sweep_vol:,.0f}, 总主动买量={total_buy_vol:,.0f}")
                print(f"    -> 结果: {results['buy_sweep_intensity']:.4f}")
        if total_sell_vol > 0:
            results['sell_sweep_intensity'] = sell_sweep_vol / total_sell_vol
            if enable_probe and is_target_date:
                print(f"--- [探针 ASM.{trade_date_str}] sell_sweep_intensity ---")
                print(f"    - 原料: 卖方扫单量={sell_sweep_vol:,.0f}, 总主动卖量={total_sell_vol:,.0f}")
                print(f"    -> 结果: {results['sell_sweep_intensity']:.4f}")
        return results

    @staticmethod
    def _calculate_vpin(context: dict) -> dict:
        """计算VPIN (Volume-Synchronized Probability of Informed Trading)"""
        tick_df = context.get('tick_df')
        total_volume = context.get('total_volume_safe')
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        results = {'vpin_score': np.nan}
        if tick_df is None or tick_df.empty or total_volume == 0:
            return results
        vpin_bucket_size = total_volume / 50
        vpin_window = 10
        if vpin_bucket_size > 0:
            tick_df['buy_vol'] = np.where(tick_df['type'] == 'B', tick_df['volume'], 0)
            tick_df['sell_vol'] = np.where(tick_df['type'] == 'S', tick_df['volume'], 0)
            tick_df['cum_vol'] = tick_df['volume'].cumsum()
            tick_df['bucket'] = (tick_df['cum_vol'] // vpin_bucket_size).astype(int)
            bucket_imbalance = tick_df.groupby('bucket').agg(buy_vol=('buy_vol', 'sum'), sell_vol=('sell_vol', 'sum'))
            bucket_imbalance['imbalance'] = bucket_imbalance['buy_vol'] - bucket_imbalance['sell_vol']
            if len(bucket_imbalance) > vpin_window:
                imbalance_std = bucket_imbalance['imbalance'].rolling(window=vpin_window).std().bfill()
                abs_imbalance = bucket_imbalance['imbalance'].abs()
                sigma_imbalance = imbalance_std.replace(0, np.nan)
                z_score = abs_imbalance / sigma_imbalance
                vpin_series = z_score.apply(lambda z: norm.cdf(z) if pd.notna(z) else np.nan)
                results['vpin_score'] = vpin_series.mean()
                if enable_probe and is_target_date:
                    print(f"--- [探针 ASM.{trade_date_str}] vpin_score ---")
                    print(f"    - 原料: 分桶数量={len(bucket_imbalance)}, 每桶容量={vpin_bucket_size:,.0f}")
                    print(f"    - 节点: 平均订单失衡绝对值={abs_imbalance.mean():,.0f}, 平均失衡标准差={imbalance_std.mean():,.0f}")
                    print(f"    -> 结果: {results['vpin_score']:.4f}")
        return results

    @staticmethod
    def _calculate_hf_mechanics(context: dict) -> dict:
        """
        【V30.19 · 微观内核索引访问修复】
        - 核心修复: 修正了因 trade_time 已被设为索引而导致的KeyError。
        - 解决方案: 修改 iterrows() 循环以直接捕获索引作为时间戳，而非在行数据中查找列。
        """
        tick_df = context.get('tick_df')
        group = context.get('group')
        daily_series_for_day = context.get('daily_series_for_day')
        atr_14 = context.get('atr_14')
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        results = {
            'active_volume_price_efficiency': np.nan,
            'absorption_strength_index': np.nan,
            'distribution_pressure_index': np.nan,
        }
        if tick_df is None or tick_df.empty or group is None or group.empty:
            return results
        total_buy_vol = tick_df[tick_df['type'] == 'B']['volume'].sum()
        total_sell_vol = tick_df[tick_df['type'] == 'S']['volume'].sum()
        total_volume = total_buy_vol + total_sell_vol + tick_df[tick_df['type'] == 'M']['volume'].sum()
        day_open_qfq = daily_series_for_day.get('open_qfq')
        day_close_qfq = daily_series_for_day.get('close_qfq')
        net_active_volume = total_buy_vol - total_sell_vol
        price_change_in_atr = (day_close_qfq - day_open_qfq) / atr_14 if pd.notna(atr_14) and atr_14 > 0 else 0
        if net_active_volume != 0 and total_volume > 0:
            results['active_volume_price_efficiency'] = price_change_in_atr / (net_active_volume / total_volume)
        tick_df.index = tick_df.index.tz_convert('Asia/Shanghai')
        down_minutes_df = group[group['close'] < group['open']]
        up_minutes_df = group[group['close'] > group['open']]
        if not down_minutes_df.empty:
            active_buy_on_dip, active_sell_on_dip = 0, 0
            # 直接从iterrows()的索引中获取minute_start
            for minute_start, minute_row in down_minutes_df.iterrows():
                minute_end = minute_start + pd.Timedelta(minutes=1)
                ticks_in_minute = tick_df[(tick_df.index >= minute_start) & (tick_df.index < minute_end)]
                active_buy_on_dip += ticks_in_minute[ticks_in_minute['type'] == 'B']['volume'].sum()
                active_sell_on_dip += ticks_in_minute[ticks_in_minute['type'] == 'S']['volume'].sum()
            if active_sell_on_dip > 0:
                results['absorption_strength_index'] = active_buy_on_dip / active_sell_on_dip
                if enable_probe and is_target_date:
                    print(f"--- [探针 ASM.{trade_date_str}] absorption_strength_index ---")
                    print(f"    - 原料: 下跌分钟内主动买量={active_buy_on_dip:,.0f}, 下跌分钟内主动卖量={active_sell_on_dip:,.0f}")
                    print(f"    -> 结果: {results['absorption_strength_index']:.4f}")
        if not up_minutes_df.empty:
            active_sell_on_rally, active_buy_on_rally = 0, 0
            # 直接从iterrows()的索引中获取minute_start
            for minute_start, minute_row in up_minutes_df.iterrows():
                minute_end = minute_start + pd.Timedelta(minutes=1)
                ticks_in_minute = tick_df[(tick_df.index >= minute_start) & (tick_df.index < minute_end)]
                active_sell_on_rally += ticks_in_minute[ticks_in_minute['type'] == 'S']['volume'].sum()
                active_buy_on_rally += ticks_in_minute[ticks_in_minute['type'] == 'B']['volume'].sum()
            if active_buy_on_rally > 0:
                results['distribution_pressure_index'] = active_sell_on_rally / active_buy_on_rally
                if enable_probe and is_target_date:
                    print(f"--- [探针 ASM.{trade_date_str}] distribution_pressure_index ---")
                    print(f"    - 原料: 上涨分钟内主动卖量={active_sell_on_rally:,.0f}, 上涨分钟内主动买量={active_buy_on_rally:,.0f}")
                    print(f"    -> 结果: {results['distribution_pressure_index']:.4f}")
        return results

    @staticmethod
    def _calculate_liquidity_metrics(context: dict) -> dict:
        """
        【V27.2 · 权重对齐修复】
        - 核心修复: 修复了因数值列表与权重列表长度不匹配导致的 np.average TypeError。
        - 解决方案: 在循环内同步构建数值列表和其对应的权重列表，确保二者长度和顺序严格一致。
        """
        tick_df = context.get('tick_df')
        level5_df = context.get('level5_df')
        realtime_df = context.get('realtime_df')
        daily_series_for_day = context.get('daily_series_for_day')
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        results = {
            'market_impact_cost': np.nan,
            'liquidity_slope': np.nan,
            'liquidity_authenticity_score': np.nan,
        }
        if tick_df is None or tick_df.empty or level5_df is None or level5_df.empty:
            return results
        column_rename_map = {**{f'buy_price{i}': f'b{i}_p' for i in range(1, 6)}, **{f'buy_volume{i}': f'b{i}_v' for i in range(1, 6)}, **{f'sell_price{i}': f'a{i}_p' for i in range(1, 6)}, **{f'sell_volume{i}': f'a{i}_v' for i in range(1, 6)}}
        level5_df_renamed = level5_df.copy().rename(columns=column_rename_map)
        if realtime_df is not None and not realtime_df.empty:
            # 重构数据合并与循环逻辑，确保权重对齐
            snapshot_df = pd.merge_asof(realtime_df.sort_index(), level5_df_renamed.sort_index(), on='trade_time', direction='backward')
            snapshot_df['snapshot_volume'] = snapshot_df['volume'].diff().fillna(0).clip(lower=0)
            total_amount = daily_series_for_day.get('amount', 0)
            if total_amount > 0:
                standard_amount = float(total_amount) * 0.001
                impact_costs, weights_for_costs = [], []
                slopes, weights_for_slopes = [], []
                for _, row in snapshot_df.iterrows():
                    snapshot_volume = row['snapshot_volume']
                    if snapshot_volume <= 0:
                        continue
                    # 冲击成本计算
                    amount_to_fill, filled_amount, filled_volume = standard_amount, 0, 0
                    for i in range(1, 6):
                        price, vol = row.get(f'a{i}_p'), row.get(f'a{i}_v', 0) * 100
                        if pd.isna(price): continue
                        value = float(price) * vol
                        if amount_to_fill > value:
                            filled_amount += value; filled_volume += vol; amount_to_fill -= value
                        else:
                            filled_volume += amount_to_fill / float(price); filled_amount += amount_to_fill; break
                    if filled_volume > 0:
                        mid_price = (row.get('b1_p', 0) + row.get('a1_p', 0)) / 2
                        if mid_price > 0:
                            cost = ((filled_amount / filled_volume) / float(mid_price) - 1) * 100
                            impact_costs.append(cost)
                            weights_for_costs.append(snapshot_volume)
                    # 斜率计算
                    mid_price = (row.get('b1_p', 0) + row.get('a1_p', 0)) / 2
                    if mid_price > 0:
                        ask_x = [(float(row.get(f'a{i}_p', mid_price)) - mid_price) / mid_price for i in range(1, 6)]
                        ask_y = np.cumsum([row.get(f'a{i}_v', 0) * 100 for i in range(1, 6)])
                        if np.std(ask_x) > 0:
                            slope = linregress(ask_x, ask_y).slope
                            slopes.append(slope)
                            weights_for_slopes.append(snapshot_volume)
                if impact_costs and sum(weights_for_costs) > 0:
                    results['market_impact_cost'] = np.average(impact_costs, weights=weights_for_costs)
                if slopes and sum(weights_for_slopes) > 0:
                    results['liquidity_slope'] = np.average(slopes, weights=weights_for_slopes)
        return results

    @staticmethod
    def _calculate_vwap_reversion(context: dict) -> dict:
        """计算VWAP均值回归相关性"""
        minute_df = context.get('continuous_group')
        results = {'vwap_mean_reversion_corr': np.nan}
        if minute_df is not None and not minute_df.empty and 'minute_vwap' in minute_df.columns and len(minute_df) > 1:
            daily_vwap = (minute_df['amount'].sum() / minute_df['vol'].sum()) if minute_df['vol'].sum() > 0 else np.nan
            if pd.notna(daily_vwap):
                deviation = minute_df['minute_vwap'] - daily_vwap
                results['vwap_mean_reversion_corr'] = deviation.autocorr(lag=1)
        return results
