# services/microstructure_dynamics_calculators.py
import pandas as pd
import numpy as np
from scipy.stats import norm, linregress
import numba # 确保已导入
from typing import Tuple

@numba.njit(cache=True)
def _numba_calculate_ofi_static_dynamic(
    buy_price1_arr: np.ndarray, buy_volume1_arr: np.ndarray,
    sell_price1_arr: np.ndarray, sell_volume1_arr: np.ndarray,
    prev_buy_price1_arr: np.ndarray, prev_buy_volume1_arr: np.ndarray,
    prev_sell_price1_arr: np.ndarray, prev_sell_volume1_arr: np.ndarray
) -> np.ndarray:
    """
    【Numba优化版】计算订单流失衡 (OFI) 的静态和动态部分。
    """
    n = len(buy_price1_arr)
    ofi_series = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i == 0: # 第一个元素没有前一个状态，OFI为0
            continue
        delta_buy_price = buy_price1_arr[i] - prev_buy_price1_arr[i]
        delta_sell_price = sell_price1_arr[i] - prev_sell_price1_arr[i]
        ofi_static = 0.0
        if delta_buy_price == 0 and delta_sell_price == 0:
            ofi_static = buy_volume1_arr[i] - prev_buy_volume1_arr[i] # 假设买一价和卖一价不变时，OFI由买一量变化决定
        ofi_dynamic = 0.0
        if delta_buy_price > 0:
            ofi_dynamic += prev_buy_volume1_arr[i] # 买一价上涨，前一刻的买一量被吃掉
        elif delta_buy_price < 0:
            ofi_dynamic -= buy_volume1_arr[i] # 买一价下跌，当前买一量是新的
        if delta_sell_price > 0:
            ofi_dynamic += sell_volume1_arr[i] # 卖一价上涨，当前卖一量是新的
        elif delta_sell_price < 0:
            ofi_dynamic -= prev_sell_volume1_arr[i] # 卖一价下跌，前一刻的卖一量被吃掉
        ofi_series[i] = ofi_static + ofi_dynamic
    return ofi_series

@numba.njit(cache=True)
def _numba_calculate_vpin_buckets(
    cum_vol_arr: np.ndarray, volume_arr: np.ndarray,
    buy_vol_arr: np.ndarray, sell_vol_arr: np.ndarray,
    vpin_bucket_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    【Numba优化版】计算VPIN桶内的买卖失衡。
    返回每个桶的失衡值和桶的索引。
    """
    # 明确定义空数组，以帮助Numba进行类型推断
    empty_float_array = np.empty(0, dtype=np.float64)
    empty_int_array = np.empty(0, dtype=np.int64)

    if vpin_bucket_size <= 0:
        return empty_float_array, empty_int_array
    
    n = len(cum_vol_arr)
    # 防御性检查：如果传入的累计成交量数组为空，直接返回空数组
    if n == 0:
        return empty_float_array, empty_int_array

    # 预估最大桶数，避免动态列表增长开销
    max_buckets = int(cum_vol_arr[-1] / vpin_bucket_size) + 2
    
    # 防御性检查：如果计算出的最大桶数无效，直接返回空数组
    if max_buckets <= 0:
        return empty_float_array, empty_int_array

    bucket_imbalance = np.zeros(max_buckets, dtype=np.float64)
    bucket_buy_vol = np.zeros(max_buckets, dtype=np.float64)
    bucket_sell_vol = np.zeros(max_buckets, dtype=np.float64)
    current_bucket_idx = 0

    for i in range(n):
        bucket_idx = int(cum_vol_arr[i] / vpin_bucket_size)
        # 确保桶索引在范围内
        if bucket_idx >= max_buckets:
            # 如果超出预估范围，跳过此元素，避免索引越界
            continue 
        bucket_buy_vol[bucket_idx] += buy_vol_arr[i]
        bucket_sell_vol[bucket_idx] += sell_vol_arr[i]
        current_bucket_idx = max(current_bucket_idx, bucket_idx)
    # 截取实际使用的桶
    actual_buckets = current_bucket_idx + 1
    # 防御性检查：如果实际桶数为0或负数，返回空数组
    if actual_buckets <= 0:
        return empty_float_array, empty_int_array

    imbalance_values = bucket_buy_vol[:actual_buckets] - bucket_sell_vol[:actual_buckets]
    bucket_indices = np.arange(actual_buckets)
    return imbalance_values, bucket_indices

@numba.njit(cache=True)
def _numba_calculate_active_volume_price_efficiency(
    price_arr: np.ndarray, volume_arr: np.ndarray, price_change_arr: np.ndarray
) -> float:
    """
    【Numba优化版】计算累计推力与累计价格位移的相关性。
    """
    n = len(price_arr)
    if n < 2:
        return np.nan
    cum_thrust = np.zeros(n, dtype=np.float64)
    cum_price_change = np.zeros(n, dtype=np.float64)
    first_price = price_arr[0]
    for i in range(n):
        # 计算每笔tick的有效推力
        net_thrust_volume = volume_arr[i] * np.sign(price_change_arr[i])
        if i == 0:
            cum_thrust[i] = net_thrust_volume
            cum_price_change[i] = price_arr[i] - first_price
        else:
            cum_thrust[i] = cum_thrust[i-1] + net_thrust_volume
            cum_price_change[i] = price_arr[i] - first_price
    # 计算相关性
    # Numba 0.58+ 支持 np.corrcoef，但为了更广泛的兼容性，手动实现
    mean_cum_thrust = np.mean(cum_thrust)
    mean_cum_price_change = np.mean(cum_price_change)
    numerator = np.sum((cum_thrust - mean_cum_thrust) * (cum_price_change - mean_cum_price_change))
    denominator_thrust = np.sqrt(np.sum((cum_thrust - mean_cum_thrust)**2))
    denominator_price = np.sqrt(np.sum((cum_price_change - mean_cum_price_change)**2))
    denominator = denominator_thrust * denominator_price
    if denominator == 0:
        return 0.0 # 如果其中一个序列没有变化，相关性为0
    return numerator / denominator

@numba.njit(cache=True)
def _numba_calculate_liquidity_authenticity_score(
    buy_price1_arr: np.ndarray, buy_volume1_arr: np.ndarray,
    sell_price1_arr: np.ndarray, sell_volume1_arr: np.ndarray,
    tick_prices_arr: np.ndarray, tick_times_arr: np.ndarray,
    level5_times_arr: np.ndarray,
    buy_commitment_threshold: float, sell_commitment_threshold: float
) -> Tuple[int, int]:
    """
    【Numba优化版】计算流动性承诺-兑现分数。
    """
    fulfillments = 0
    defaults = 0
    n_level5 = len(buy_price1_arr)
    n_tick = len(tick_prices_arr)
    # 识别买方承诺（大额买单出现）
    for i in range(n_level5):
        if buy_volume1_arr[i] > buy_commitment_threshold:
            # 检查是否是新增的大单（与前一刻相比）
            if i > 0 and buy_volume1_arr[i] > buy_volume1_arr[i-1] * 2: # 简化判断为显著增加
                commit_price = buy_price1_arr[i]
                # 追踪此承诺未来20个快照
                future_snapshots_start_idx = i + 1
                future_snapshots_end_idx = min(n_level5, future_snapshots_start_idx + 20)
                pressure_found = False
                for j in range(future_snapshots_start_idx, future_snapshots_end_idx):
                    if sell_price1_arr[j] <= commit_price + 0.02: # 卖一价接近承诺价
                        pressure_found = True
                        
                        # 结局判断：是成交了还是撤单了？
                        if buy_volume1_arr[j] < buy_volume1_arr[i] * 0.5: # 大单在压力下消失
                            defaults += 1
                        else:
                            # 检查是否有真实成交 (简化为tick数据中是否有承诺价的成交)
                            # 找到level5快照时间对应的tick数据范围
                            level5_time_start = level5_times_arr[i]
                            level5_time_end = level5_times_arr[j]
                            
                            tick_start_idx = np.searchsorted(tick_times_arr, level5_time_start)
                            tick_end_idx = np.searchsorted(tick_times_arr, level5_time_end)
                            
                            found_trade = False
                            for k in range(tick_start_idx, tick_end_idx):
                                if tick_prices_arr[k] == commit_price:
                                    found_trade = True
                                    break
                            
                            if found_trade:
                                fulfillments += 1
                        break # 找到压力点后就停止追踪
                # 如果在未来20个快照内没有找到压力点，也算作违约（承诺未被测试）
                if not pressure_found:
                    defaults += 1
    # 识别卖方承诺（大额卖单出现）
    for i in range(n_level5):
        if sell_volume1_arr[i] > sell_commitment_threshold:
            if i > 0 and sell_volume1_arr[i] > sell_volume1_arr[i-1] * 2: # 简化判断为显著增加
                commit_price = sell_price1_arr[i]
                future_snapshots_start_idx = i + 1
                future_snapshots_end_idx = min(n_level5, future_snapshots_start_idx + 20)
                pressure_found = False
                for j in range(future_snapshots_start_idx, future_snapshots_end_idx):
                    if buy_price1_arr[j] >= commit_price - 0.02: # 买一价接近承诺价
                        pressure_found = True
                        
                        if sell_volume1_arr[j] < sell_volume1_arr[i] * 0.5:
                            defaults += 1
                        else:
                            level5_time_start = level5_times_arr[i]
                            level5_time_end = level5_times_arr[j]
                            
                            tick_start_idx = np.searchsorted(tick_times_arr, level5_time_start)
                            tick_end_idx = np.searchsorted(tick_times_arr, level5_time_end)
                            
                            found_trade = False
                            for k in range(tick_start_idx, tick_end_idx):
                                if tick_prices_arr[k] == commit_price:
                                    found_trade = True
                                    break
                            
                            if found_trade:
                                fulfillments += 1
                        break
                if not pressure_found:
                    defaults += 1
                    
    return fulfillments, defaults

@numba.njit(cache=True)
def _numba_calculate_vwap_reversion_corr(deviation_arr: np.ndarray) -> float:
    """
    【Numba优化版】计算VWAP均值回归相关性。
    """
    n = len(deviation_arr)
    if n < 2:
        return np.nan
    # 计算自相关系数 (lag=1)
    # corr(X_t, X_{t-1}) = cov(X_t, X_{t-1}) / (std(X_t) * std(X_{t-1}))
    # 移除NaN
    clean_deviation = deviation_arr[~np.isnan(deviation_arr)]
    if len(clean_deviation) < 2:
        return np.nan
    x_t = clean_deviation[1:]
    x_t_minus_1 = clean_deviation[:-1]
    if np.std(x_t) == 0 or np.std(x_t_minus_1) == 0:
        return 0.0 # 如果序列没有变化，自相关为0
    return np.corrcoef(x_t, x_t_minus_1)[0, 1]

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
            df_prev = df.shift(1).fillna(0) # 填充NaN以避免Numba处理NaN
            # 提取NumPy数组
            buy_price1_arr = df['buy_price1'].values
            buy_volume1_arr = df['buy_volume1'].values
            sell_price1_arr = df['sell_price1'].values
            sell_volume1_arr = df['sell_volume1'].values
            prev_buy_price1_arr = df_prev['buy_price1'].values
            prev_buy_volume1_arr = df_prev['buy_volume1'].values
            prev_sell_price1_arr = df_prev['sell_price1'].values
            prev_sell_volume1_arr = df_prev['sell_volume1'].values
            # 调用Numba优化函数
            ofi_series_numba = _numba_calculate_ofi_static_dynamic(
                buy_price1_arr, buy_volume1_arr,
                sell_price1_arr, sell_volume1_arr,
                prev_buy_price1_arr, prev_buy_volume1_arr,
                prev_sell_price1_arr, prev_sell_volume1_arr
            )
            total_ofi = np.nansum(ofi_series_numba)
            if total_volume > 0:
                results['order_flow_imbalance_score'] = total_ofi / total_volume
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
        if total_sell_vol > 0:
            results['sell_sweep_intensity'] = sell_sweep_vol / total_sell_vol
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
            # 提取NumPy数组
            cum_vol_arr = tick_df['cum_vol'].values
            volume_arr = tick_df['volume'].values
            buy_vol_arr = tick_df['buy_vol'].values
            sell_vol_arr = tick_df['sell_vol'].values
            # 调用Numba优化函数
            imbalance_values, bucket_indices = _numba_calculate_vpin_buckets(
                cum_vol_arr, volume_arr, buy_vol_arr, sell_vol_arr, vpin_bucket_size
            )
            if len(imbalance_values) > vpin_window:
                # 将Numba结果转换回Pandas Series进行后续滚动计算
                bucket_imbalance_series = pd.Series(imbalance_values, index=bucket_indices)
                imbalance_std = bucket_imbalance_series.rolling(window=vpin_window).std().bfill()
                abs_imbalance = bucket_imbalance_series.abs()
                sigma_imbalance = imbalance_std.replace(0, np.nan)
                z_score = abs_imbalance / sigma_imbalance
                vpin_series = z_score.apply(lambda z: norm.cdf(z) if pd.notna(z) else np.nan)
                results['vpin_score'] = vpin_series.mean()
        return results

    @staticmethod
    def _calculate_hf_mechanics(context: dict) -> dict:
        """
        【V61.0 · 博弈精研】
        - `active_volume_price_efficiency` 升维: 逻辑彻底重构。不再计算静态的“终局”比值，
                     而是通过计算日内“累计推力”与“累计价格位移”两条曲线的相关系数，
                     来动态追溯推力的“过程有效性”，洞察主力资金的控盘合力。
        - 核心优化: 使用Numba优化后的相关性计算函数。
        """
        tick_df = context.get('tick_df')
        debug_info = context.get('debug', {})
        is_target_date = debug_info.get('is_target_date', False)
        enable_probe = debug_info.get('enable_probe', False)
        trade_date_str = debug_info.get('trade_date_str', 'N/A')
        results = {
            'active_volume_price_efficiency': np.nan,
        }
        if tick_df is None or tick_df.empty or len(tick_df) < 2:
            return results
        # 1. 计算每笔tick的有效推力
        price_arr = tick_df['price'].values
        volume_arr = tick_df['volume'].values
        # 确保 price_change 存在且有效
        price_change_arr = np.zeros_like(price_arr, dtype=np.float64)
        if 'price_change' in tick_df.columns and not tick_df['price_change'].isnull().all():
            self_calculated_change = np.diff(price_arr, prepend=price_arr[0]) # 计算实际价格变化
            zero_change_mask = (tick_df['price_change'].values == 0)
            price_change_arr = np.where(zero_change_mask, self_calculated_change, tick_df['price_change'].values)
        else: # 回退逻辑，直接使用价格变化
            price_change_arr = np.diff(price_arr, prepend=price_arr[0])
        # 调用Numba优化函数
        correlation = _numba_calculate_active_volume_price_efficiency(
            price_arr, volume_arr, price_change_arr
        )
        results['active_volume_price_efficiency'] = correlation
        return results

    @staticmethod
    def _calculate_liquidity_metrics(context: dict) -> dict:
        """
        【V70.0 · 流动性验真】
        - 核心升维: 彻底重构 `liquidity_authenticity_score`。不再依赖静态盘口形态，而是
                     引入“流动性承诺-兑现”动态追踪模型。通过识别盘口异常大额挂单，并追踪
                     其在价格压力下的最终结局（真实成交或提前撤单），深度量化挂单的“诚意”，
                     从而辨别“铁壁”与“幻象”。
        - 核心优化: 使用Numba优化后的流动性承诺-兑现分数计算函数。
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
        if tick_df is None or tick_df.empty or level5_df is None or level5_df.empty or len(level5_df) < 2:
            return results
        # --- 保留 market_impact_cost 和 liquidity_slope 的计算逻辑 ---
        column_rename_map = {**{f'buy_price{i}': f'b{i}_p' for i in range(1, 6)}, **{f'buy_volume{i}': f'b{i}_v' for i in range(1, 6)}, **{f'sell_price{i}': f'a{i}_p' for i in range(1, 6)}, **{f'sell_volume{i}': f'a{i}_v' for i in range(1, 6)}}
        level5_df_renamed = level5_df.copy().rename(columns=column_rename_map)
        if realtime_df is not None and not realtime_df.empty:
            snapshot_df = pd.merge_asof(realtime_df.sort_index(), level5_df_renamed.sort_index(), on='trade_time', direction='backward')
            snapshot_df['snapshot_volume'] = snapshot_df['volume'].diff().fillna(0).clip(lower=0)
            total_amount = daily_series_for_day.get('amount', 0)
            if total_amount > 0:
                standard_amount = float(total_amount) * 0.001
                impact_costs, weights_for_costs = [], []
                slopes, weights_for_slopes = [], []
                for _, row in snapshot_df.iterrows():
                    snapshot_volume = row['snapshot_volume']
                    if snapshot_volume <= 0: continue
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
        # --- 新增 `liquidity_authenticity_score` 的升维计算逻辑 ---
        # 提取NumPy数组
        buy_price1_arr = level5_df['buy_price1'].values
        buy_volume1_arr = level5_df['buy_volume1'].values
        sell_price1_arr = level5_df['sell_price1'].values
        sell_volume1_arr = level5_df['sell_volume1'].values
        tick_prices_arr = tick_df['price'].values
        tick_times_arr = tick_df.index.values.astype(np.int64) # 将Timestamp转换为int64
        level5_times_arr = level5_df.index.values.astype(np.int64) # 将Timestamp转换为int64
        b1_vol_mean, b1_vol_std = buy_volume1_arr.mean(), buy_volume1_arr.std()
        a1_vol_mean, a1_vol_std = sell_volume1_arr.mean(), sell_volume1_arr.std()
        buy_commitment_threshold = b1_vol_mean + 2 * b1_vol_std
        sell_commitment_threshold = a1_vol_mean + 2 * a1_vol_std
        # 调用Numba优化函数
        fulfillments, defaults = _numba_calculate_liquidity_authenticity_score(
            buy_price1_arr, buy_volume1_arr,
            sell_price1_arr, sell_volume1_arr,
            tick_prices_arr, tick_times_arr,
            level5_times_arr,
            buy_commitment_threshold, sell_commitment_threshold
        )
        total_events = fulfillments + defaults
        if total_events > 0:
            results['liquidity_authenticity_score'] = fulfillments / total_events
        else:
            results['liquidity_authenticity_score'] = 0.5 # 无事件发生，给予中性分
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
                # 提取NumPy数组
                deviation_arr = deviation.values
                # 调用Numba优化函数
                results['vwap_mean_reversion_corr'] = _numba_calculate_vwap_reversion_corr(deviation_arr)
        return results


