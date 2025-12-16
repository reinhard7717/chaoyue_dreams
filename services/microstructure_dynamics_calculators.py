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
        【V61.0 · 博弈精研】
        - `active_volume_price_efficiency` 升维: 逻辑彻底重构。不再计算静态的“终局”比值，
                     而是通过计算日内“累计推力”与“累计价格位移”两条曲线的相关系数，
                     来动态追溯推力的“过程有效性”，洞察主力资金的控盘合力。
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
        # 升维：计算累计推力与累计价格位移的相关性
        # 1. 计算每笔tick的有效推力
        if 'price_change' in tick_df.columns and not tick_df['price_change'].isnull().all():
            self_calculated_change = tick_df['price'].diff().fillna(0)
            zero_change_mask = tick_df['price_change'] == 0
            effective_price_change = np.where(zero_change_mask, self_calculated_change, tick_df['price_change'])
            net_thrust_volume = tick_df['volume'] * np.sign(effective_price_change)
        else: # 回退逻辑
            buy_vol = np.where(tick_df['type'] == 'B', tick_df['volume'], 0)
            sell_vol = np.where(tick_df['type'] == 'S', -tick_df['volume'], 0)
            net_thrust_volume = buy_vol + sell_vol
        # 2. 构建两条累计曲线
        tick_df['cum_thrust'] = net_thrust_volume.cumsum()
        first_price = tick_df['price'].iloc[0]
        tick_df['cum_price_change'] = tick_df['price'] - first_price
        # 3. 按分钟重采样以进行相关性分析（避免tick级别噪声过大）
        resampled_df = tick_df[['cum_thrust', 'cum_price_change']].resample('1min').last().dropna()
        if len(resampled_df) > 2:
            correlation = resampled_df['cum_thrust'].corr(resampled_df['cum_price_change'])
            results['active_volume_price_efficiency'] = correlation
            if enable_probe and is_target_date:
                print(f"--- [探针 ASM.{trade_date_str}] active_volume_price_efficiency (博弈) ---")
                print(f"    - 模式: 过程追溯 (高频)")
                print(f"    - 原料: {len(resampled_df)}个分钟采样点上的“累计推力”与“累计价格位移”序列")
                print(f"    - 计算: corr(cum_thrust, cum_price_change)")
                print(f"    -> 结果: {results.get('active_volume_price_efficiency', np.nan):.4f}")
        return results
    @staticmethod
    def _calculate_liquidity_metrics(context: dict) -> dict:
        """
        【V70.0 · 流动性验真】
        - 核心升维: 彻底重构 `liquidity_authenticity_score`。不再依赖静态盘口形态，而是
                     引入“流动性承诺-兑现”动态追踪模型。通过识别盘口异常大额挂单，并追踪
                     其在价格压力下的最终结局（真实成交或提前撤单），深度量化挂单的“诚意”，
                     从而辨别“铁壁”与“幻象”。
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
        df = level5_df[['buy_price1', 'buy_volume1', 'sell_price1', 'sell_volume1']].copy()
        df['prev_b1_v'] = df['buy_volume1'].shift(1)
        df['prev_a1_v'] = df['sell_volume1'].shift(1)
        # 1. 定义“大单”阈值
        b1_vol_mean, b1_vol_std = df['buy_volume1'].mean(), df['buy_volume1'].std()
        a1_vol_mean, a1_vol_std = df['sell_volume1'].mean(), df['sell_volume1'].std()
        buy_commitment_threshold = b1_vol_mean + 2 * b1_vol_std
        sell_commitment_threshold = a1_vol_mean + 2 * a1_vol_std
        # 2. 识别“承诺”与“结局”
        fulfillments = 0
        defaults = 0
        # 识别买方承诺（大额买单出现）
        buy_commitments = df[(df['buy_volume1'] > buy_commitment_threshold) & (df['buy_volume1'] > df['prev_b1_v'] * 2)]
        for idx, commit in buy_commitments.iterrows():
            # 追踪此承诺
            future_snapshots = df.loc[idx:].iloc[1:21] # 观察未来20个快照（约1分钟）
            if future_snapshots.empty: continue
            commit_price = commit['buy_price1']
            # 压力测试：卖一价是否接近承诺价
            pressure_snapshots = future_snapshots[future_snapshots['sell_price1'] <= commit_price + 0.02]
            if not pressure_snapshots.empty:
                first_pressure_point = pressure_snapshots.iloc[0]
                # 结局判断：是成交了还是撤单了？
                if first_pressure_point['buy_volume1'] < commit['buy_volume1'] * 0.5:
                    defaults += 1 # 大单在压力下消失，视为违约
                else:
                    # 检查是否有真实成交
                    related_ticks = tick_df.loc[idx:first_pressure_point.name]
                    if not related_ticks.empty and (related_ticks['price'] == commit_price).any():
                        fulfillments += 1 # 有成交，视为兑现
        # 识别卖方承诺（大额卖单出现）
        sell_commitments = df[(df['sell_volume1'] > sell_commitment_threshold) & (df['sell_volume1'] > df['prev_a1_v'] * 2)]
        for idx, commit in sell_commitments.iterrows():
            future_snapshots = df.loc[idx:].iloc[1:21]
            if future_snapshots.empty: continue
            commit_price = commit['sell_price1']
            pressure_snapshots = future_snapshots[future_snapshots['buy_price1'] >= commit_price - 0.02]
            if not pressure_snapshots.empty:
                first_pressure_point = pressure_snapshots.iloc[0]
                if first_pressure_point['sell_volume1'] < commit['sell_volume1'] * 0.5:
                    defaults += 1
                else:
                    related_ticks = tick_df.loc[idx:first_pressure_point.name]
                    if not related_ticks.empty and (related_ticks['price'] == commit_price).any():
                        fulfillments += 1
        # 3. 计算最终得分
        total_events = fulfillments + defaults
        if total_events > 0:
            results['liquidity_authenticity_score'] = fulfillments / total_events
        else:
            results['liquidity_authenticity_score'] = 0.5 # 无事件发生，给予中性分
        if enable_probe and is_target_date:
            print(f"--- [探针 ASM.{trade_date_str}] liquidity_authenticity_score (流动性验真) ---")
            print(f"    - 原料: {len(level5_df)}个Level-5快照")
            print(f"    - 节点: 买方大单阈值={buy_commitment_threshold:,.0f}, 卖方大单阈值={sell_commitment_threshold:,.0f}")
            print(f"    - 节点: 识别到买方承诺{len(buy_commitments)}次, 卖方承诺{len(sell_commitments)}次")
            print(f"    - 节点: 承诺兑现(成交)次数={fulfillments}, 承诺违约(撤单)次数={defaults}")
            if total_events > 0:
                print(f"    - 计算: {fulfillments} / ({fulfillments} + {defaults})")
            else:
                print(f"    - 计算: 无大型挂单博弈事件，返回中性分")
            print(f"    -> 结果: {results['liquidity_authenticity_score']:.4f}")
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
