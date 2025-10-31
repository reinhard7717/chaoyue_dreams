# 文件: services/chip_feature_calculator.py

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from decimal import Decimal
import logging
logger = logging.getLogger(__name__)

class ChipFeatureCalculator:
    """
    【V11.0 · 战术中枢重构版】
    - 核心重构: 本类现在是所有筹码指标计算的唯一核心，接收原始数据并完成所有计算。
    - 逻辑注入: 新增了成交量微观结构和筹码集中度动态归因的计算逻辑。
    """
    def __init__(self, daily_chips_df: pd.DataFrame, context_data: dict):
        self.df = daily_chips_df.reset_index(drop=True)
        self.ctx = context_data
        if not self.df.empty:
            percent_sum = self.df['percent'].sum()
            if not np.isclose(percent_sum, 100.0) and percent_sum > 0:
                self.df['percent'] = (self.df['percent'] / percent_sum) * 100.0
        # 增加 prev_winner_avg_cost 用于计算获利盘行为指标
        cyq_perf_fields = [
            'total_chip_volume', 'daily_turnover_volume', 'close_price', 'high_price', 'low_price', 'prev_20d_close',
            'cost_5pct', 'cost_15pct', 'cost_50pct', 'cost_85pct', 'cost_95pct', 'weight_avg', 'winner_rate',
            'prev_concentration_90pct', 'prev_winner_avg_cost'
        ]
        
        for key in cyq_perf_fields:
            if key in self.ctx and isinstance(self.ctx[key], Decimal):
                self.ctx[key] = float(self.ctx[key])
        self._prepare_minute_data_features()

    def calculate_all_metrics(self) -> dict:
        """
        【V14.0 · 中央计算集群版】
        - 核心重构: 调用全新的、统一的计算核心 `_calculate_all_chip_features`。
        """
        # [代码修改开始]
        stock_code = self.ctx.get('stock_code', 'UNKNOWN')
        trade_date = self.ctx.get('trade_date', 'UNKNOWN')
        if self.df.empty:
            logger.warning(f"[{stock_code}] [{trade_date}] 筹码计算中止，原因：当日筹码分布数据(daily_chips_df)为空。")
            return {}
        required_keys = ['weight_avg', 'winner_rate', 'cost_95pct', 'cost_5pct', 'close_price', 'total_chip_volume']
        missing_keys = [k for k in required_keys if k not in self.ctx or self.ctx[k] is None or pd.isna(self.ctx[k])]
        if missing_keys:
            logger.warning(f"[{stock_code}] [{trade_date}] 筹码计算中止，原因：上下文(context_data)中缺少核心字段: {missing_keys}。")
            return {}
        # 步骤1: 直接从上下文中提取初始指标
        summary_info = self._get_summary_metrics_from_context()
        self.ctx.update(summary_info)
        # 步骤2: 调用统一的中央计算集群
        all_metrics = self._calculate_all_chip_features()
        # 步骤3: 计算最终的健康分 (它依赖于之前计算出的所有指标)
        health_score_info = self._calculate_chip_structure_health_score(self.ctx)
        all_metrics.update(health_score_info)
        # 步骤4: 清理临时或中间指标
        all_metrics.pop('peak_range_low', None)
        all_metrics.pop('peak_range_high', None)
        return all_metrics
        # [代码修改结束]

    def _prepare_minute_data_features(self):
        """【V2.3 · 生产就绪版】将调试用的 print 语句替换为 logger.warning。"""
        minute_df = self.ctx.get('minute_data')
        if minute_df is None or minute_df.empty:
            stock_code = self.ctx.get('stock_code', 'UNKNOWN')
            trade_date = self.ctx.get('trade_date', 'UNKNOWN')
            # [代码修改开始]
            # 核心修正：将 print 替换为 logger.warning
            logger.warning(f"[{stock_code}] [{trade_date}] 分钟数据特征准备跳过，原因：分钟数据(minute_data)为空。")
            # [代码修改结束]
            self.ctx.update({'daily_vwap': None, 'volume_above_vwap_ratio': None, 'volume_below_vwap_ratio': None})
            return
        dtype_map = {
            'amount': 'float32',
            'vol': 'int32',
            'open': 'float32',
            'close': 'float32',
            'high': 'float32',
            'low': 'float32',
        }
        for col, dtype in dtype_map.items():
            if col in minute_df.columns:
                minute_df[col] = pd.to_numeric(minute_df[col], errors='coerce').astype(dtype, errors='ignore')
        total_amount_yuan = minute_df['amount'].sum()
        total_vol_shares = minute_df['vol'].sum()
        daily_vwap = total_amount_yuan / total_vol_shares if total_vol_shares > 0 else None
        self.ctx['daily_vwap'] = daily_vwap
        if daily_vwap is None:
            self.ctx.update({'volume_above_vwap_ratio': None, 'volume_below_vwap_ratio': None})
            return
        minute_df['minute_vwap'] = (minute_df['amount'] / minute_df['vol'].replace(0, np.nan)).astype('float32', errors='ignore')
        vol_above_vwap = minute_df[minute_df['minute_vwap'] > daily_vwap]['vol'].sum()
        vol_below_vwap = minute_df[minute_df['minute_vwap'] < daily_vwap]['vol'].sum()
        total_vol = minute_df['vol'].sum()
        if total_vol > 0:
            self.ctx['volume_above_vwap_ratio'] = vol_above_vwap / total_vol
            self.ctx['volume_below_vwap_ratio'] = vol_below_vwap / total_vol
        else:
            self.ctx.update({'volume_above_vwap_ratio': 0.0, 'volume_below_vwap_ratio': 0.0})

    def _get_summary_metrics_from_context(self) -> dict:
        """【V13.0 重构】直接从上下文中提取由 cyq_perf 提供的高阶指标。"""
        weight_avg_cost = self.ctx.get('weight_avg')
        total_winner_rate = self.ctx.get('winner_rate')
        return {
            'weight_avg_cost': weight_avg_cost,
            'total_winner_rate': total_winner_rate,
        }

    def _calculate_cross_day_holder_flow(self, context: dict) -> dict:
        """
        【V1.3 · 生产就绪版】将调试用的 print 语句替换为 logger.info。
        """
        results = {
            'short_term_profit_taking_ratio': None,
            'long_term_chips_unlocked_ratio': None,
            'short_term_capitulation_ratio': None,
            'long_term_despair_selling_ratio': None,
        }
        stock_code = context.get('stock_code', 'UNKNOWN')
        trade_date = context.get('trade_date', 'UNKNOWN')
        scalar_keys = ['daily_turnover_volume', 'pre_close', 'prev_20d_close', 'total_chip_volume']
        missing_scalar_keys = [k for k in scalar_keys if context.get(k) is None or pd.isna(context.get(k))]
        if missing_scalar_keys:
            # [代码修改开始]
            # 核心修正：将 print 替换为 logger.info，因为这是计算首日预期内的行为
            logger.info(f"[{stock_code}] [{trade_date}] 跨日筹码流动计算跳过，原因：上下文缺少前一日关键标量数据: {missing_scalar_keys}。")
            # [代码修改结束]
            return results
        prev_df = context.get('prev_chip_distribution')
        if prev_df is None or prev_df.empty:
            # [代码修改开始]
            # 核心修正：将 print 替换为 logger.info
            logger.info(f"[{stock_code}] [{trade_date}] 跨日筹码流动计算跳过，原因：前一日筹码分布(prev_chip_distribution)为空或不存在。")
            # [代码修改结束]
            return results
        turnover_vol = context['daily_turnover_volume']
        prev_close = context['pre_close']
        prev_20d_close = context['prev_20d_close']
        total_chips = context['total_chip_volume']
        prev_winners_short_df = prev_df[(prev_df['price'] >= prev_20d_close) & (prev_df['price'] < prev_close)]
        prev_winners_long_df = prev_df[prev_df['price'] < prev_20d_close]
        prev_losers_short_df = prev_df[(prev_df['price'] >= prev_20d_close) & (prev_df['price'] > prev_close)]
        prev_losers_long_df = prev_df[(prev_df['price'] < prev_20d_close) & (prev_df['price'] > prev_close)]
        vol_winners_short = (prev_winners_short_df['percent'].sum() / 100) * total_chips
        vol_winners_long = (prev_winners_long_df['percent'].sum() / 100) * total_chips
        vol_losers_short = (prev_losers_short_df['percent'].sum() / 100) * total_chips
        vol_losers_long = (prev_losers_long_df['percent'].sum() / 100) * total_chips
        total_prev_percent = prev_df['percent'].sum()
        if total_prev_percent > 0:
            turnover_from_winners_short = turnover_vol * (prev_winners_short_df['percent'].sum() / total_prev_percent)
            turnover_from_winners_long = turnover_vol * (prev_winners_long_df['percent'].sum() / total_prev_percent)
            turnover_from_losers_short = turnover_vol * (prev_losers_short_df['percent'].sum() / total_prev_percent)
            turnover_from_losers_long = turnover_vol * (prev_losers_long_df['percent'].sum() / total_prev_percent)
        else:
            return results
        if vol_winners_short > 0:
            results['short_term_profit_taking_ratio'] = (turnover_from_winners_short / vol_winners_short) * 100
        if vol_winners_long > 0:
            results['long_term_chips_unlocked_ratio'] = (turnover_from_winners_long / vol_winners_long) * 100
        if vol_losers_short > 0:
            results['short_term_capitulation_ratio'] = (turnover_from_losers_short / vol_losers_short) * 100
        if vol_losers_long > 0:
            results['long_term_despair_selling_ratio'] = (turnover_from_losers_long / vol_losers_long) * 100
        return results

    def _calculate_all_chip_features(self) -> dict:
        """
        【V4.0 · 合成旅作战版】
        - 核心革命: 将所有计算象限整编为五个高度内聚的“合成旅”计算单元，实现最终的模块化。
        - 核心思想: 指挥链条达到最简化，每个“合成旅”负责一个完整的战术象限。
        """
        # [代码修改开始]
        # --- 象限一: 静态结构信号旅 ---
        static_structure_info = self._calculate_static_structure_signals()
        self.ctx.update(static_structure_info)
        # --- 象限二: 日内动态信号旅 ---
        intraday_dynamics_info = self._calculate_intraday_dynamics_signals(self.ctx)
        self.ctx.update(intraday_dynamics_info)
        # --- 象限三: 跨日流转信号旅 ---
        cross_day_flow_info = self._calculate_cross_day_flow_signals(self.ctx)
        self.ctx.update(cross_day_flow_info)
        # --- 象限四: 博弈意图信号旅 ---
        game_theoretic_info = self._calculate_game_theoretic_signals(self.ctx)
        self.ctx.update(game_theoretic_info)
        # --- 象限五: 生命体征信号旅 ---
        vital_signs_info = self._calculate_vital_sign_signals(self.ctx)
        self.ctx.update(vital_signs_info)
        # --- 整合所有指标 ---
        all_metrics = {
            **static_structure_info,
            **intraday_dynamics_info,
            **cross_day_flow_info,
            **game_theoretic_info,
            **vital_signs_info,
        }
        return all_metrics
        # [代码修改结束]

    def _calculate_static_structure_signals(self) -> dict:
        """
        【V2.0 · 战术深化版】
        - 核心升级: 引入盈利亏损质量、结构稳定性、成本分布形态三个全新的计算单元。
        - 核心思想: 从多维度对静态结构进行深度剖析，提供更具实战价值的判断依据。
        """
        results = {}
        close_price = self.ctx.get('close_price')
        # 1. 主导峰剖面 (逻辑保留)
        peaks, properties = find_peaks(self.df['percent'], prominence=0.1, width=1)
        if len(peaks) == 0 and not self.df.empty:
            peak_idx = self.df['percent'].idxmax()
            results['dominant_peak_cost'] = self.df.loc[peak_idx, 'price']
            results['dominant_peak_volume_ratio'] = self.df.loc[peak_idx, 'percent']
            results['peak_range_low'] = results['dominant_peak_cost'] * 0.995
            results['peak_range_high'] = results['dominant_peak_cost'] * 1.005
        elif len(peaks) > 0:
            main_peak_idx = peaks[np.argmax(properties['prominences'])]
            results['dominant_peak_cost'] = self.df.iloc[main_peak_idx]['price']
            results['dominant_peak_volume_ratio'] = self.df.iloc[main_peak_idx]['percent']
        if pd.notna(close_price) and results.get('dominant_peak_cost', 0) > 0:
            results['dominant_peak_profit_margin'] = (close_price / results['dominant_peak_cost'] - 1) * 100
        # 2. 集中度剖面 (逻辑保留)
        def _get_concentration(chip_df: pd.DataFrame):
            if chip_df.empty or chip_df['percent'].sum() < 1e-6: return None
            chip_df = chip_df.copy()
            chip_df['percent'] = (chip_df['percent'] / chip_df['percent'].sum()) * 100
            chip_df['cum_percent'] = chip_df['percent'].cumsum()
            avg_cost = np.average(chip_df['price'], weights=chip_df['percent'])
            if avg_cost <= 0: return None
            price_low = np.interp(5, chip_df['cum_percent'], chip_df['price'])
            price_high = np.interp(95, chip_df['cum_percent'], chip_df['price'])
            return (price_high - price_low) / avg_cost
        if pd.notna(close_price):
            results['winner_concentration_90pct'] = _get_concentration(self.df[self.df['price'] < close_price])
            results['loser_concentration_90pct'] = _get_concentration(self.df[self.df['price'] > close_price])
        # 3. 动态压力支撑 (逻辑保留)
        atr_14d = self.ctx.get('atr_14d')
        if pd.notna(close_price) and pd.notna(atr_14d) and atr_14d > 0:
            zone_width = 0.5 * atr_14d
            pressure_df = self.df[(self.df['price'] > close_price) & (self.df['price'] <= close_price + zone_width)]
            support_df = self.df[(self.df['price'] >= close_price - zone_width) & (self.df['price'] < close_price)]
            results['potential_pressure_pct'] = pressure_df['percent'].sum()
            results['potential_support_pct'] = support_df['percent'].sum()
        # 4. 盈亏结构 (逻辑保留)
        if pd.notna(close_price):
            winners_df = self.df[self.df['price'] < close_price]
            losers_df = self.df[self.df['price'] > close_price]
            results['total_winner_rate'] = winners_df['percent'].sum()
            results['total_loser_rate'] = losers_df['percent'].sum()
            if not winners_df.empty and winners_df['percent'].sum() > 0:
                winner_avg_cost = np.average(winners_df['price'], weights=winners_df['percent'])
                if winner_avg_cost > 0: results['winner_profit_margin_avg'] = (close_price / winner_avg_cost - 1) * 100
            if not losers_df.empty and losers_df['percent'].sum() > 0:
                loser_avg_cost = np.average(losers_df['price'], weights=losers_df['percent'])
                if loser_avg_cost > 0: results['loser_loss_margin_avg'] = (close_price / loser_avg_cost - 1) * 100
            high_20d, low_20d = self.ctx.get('high_20d'), self.ctx.get('low_20d')
            if pd.notna(high_20d) and pd.notna(low_20d):
                active_mask = (self.df['price'] >= low_20d) & (self.df['price'] <= high_20d)
                results['active_winner_rate'] = self.df[active_mask & (self.df['price'] < close_price)]['percent'].sum()
                results['active_loser_rate'] = self.df[active_mask & (self.df['price'] > close_price)]['percent'].sum()
                results['locked_profit_rate'] = self.df[~active_mask & (self.df['price'] < close_price)]['percent'].sum()
                results['locked_loss_rate'] = self.df[~active_mask & (self.df['price'] > close_price)]['percent'].sum()
        # 5. 断层动态 (逻辑保留)
        peak_cost = results.get('dominant_peak_cost')
        if pd.notna(peak_cost) and pd.notna(close_price) and pd.notna(atr_14d) and atr_14d > 0:
            magnitude = (close_price - peak_cost) / atr_14d
            results['chip_fault_magnitude'] = magnitude
            fault_zone_low, fault_zone_high = sorted([peak_cost, close_price])
            fault_zone_df = self.df[(self.df['price'] > fault_zone_low) & (self.df['price'] < fault_zone_high)]
            if not fault_zone_df.empty: results['chip_fault_blockage_ratio'] = fault_zone_df['percent'].sum()
        # 6. 分层成本 (逻辑保留)
        high_20d, low_20d = self.ctx.get('high_20d'), self.ctx.get('low_20d')
        if pd.notna(high_20d) and pd.notna(low_20d):
            active_zone_mask = (self.df['price'] >= low_20d) & (self.df['price'] <= high_20d)
            short_term_chips_df = self.df[active_zone_mask]
            long_term_chips_df = self.df[~active_zone_mask]
            if not short_term_chips_df.empty and short_term_chips_df['percent'].sum() > 0:
                results['short_term_holder_cost'] = np.average(short_term_chips_df['price'], weights=short_term_chips_df['percent'])
            if not long_term_chips_df.empty and long_term_chips_df['percent'].sum() > 0:
                results['long_term_holder_cost'] = np.average(long_term_chips_df['price'], weights=long_term_chips_df['percent'])
        # [代码修改开始]
        # --- 7. 新增: 盈利亏损质量分析 ---
        pl_quality_metrics = self._calculate_profit_loss_quality(self.df, close_price)
        results.update(pl_quality_metrics)
        # --- 8. 新增: 结构稳定性评估 ---
        # 注意: 此方法依赖于之前计算出的指标，因此放在后面
        stability_metrics = self._calculate_structural_stability(results)
        results.update(stability_metrics)
        # --- 9. 新增: 成本分布形态分析 ---
        distribution_stats = self._calculate_cost_distribution_statistics(self.df)
        results.update(distribution_stats)
        # [代码修改结束]
        return results

    def _calculate_game_theoretic_signals(self, context: dict) -> dict:
        """
        【V1.0 · 新增 · 博弈意图信号旅】
        - 核心革命: 整合“博弈意图”象限中所有计算单元，形成统一的博弈意图计算引擎。
        - 包含逻辑: 战术序列流、高级结构、获利盘信念、以及所有原 `_calculate_intent_signals` 中的逻辑。
        """
        # [代码新增开始]
        results = {}
        minute_df = context.get('minute_data')
        total_daily_vol = context.get('daily_turnover_volume')
        close_price = context.get('close_price')
        # 1. 战术序列流 (Intraday Tactical Flow)
        required_cols_1 = ['minute_vwap', 'main_force_net_vol', 'vol']
        if minute_df is not None and not minute_df.empty and pd.notna(total_daily_vol) and total_daily_vol > 0 and all(c in minute_df.columns for c in required_cols_1):
            peaks, _ = find_peaks(minute_df['minute_vwap'], prominence=0.001)
            troughs, _ = find_peaks(-minute_df['minute_vwap'], prominence=0.001)
            turning_points = sorted(list(set(np.concatenate(([0], troughs, peaks, [len(minute_df)-1])))))
            suppressive_vol, rally_dist_vol, rally_acc_vol, panic_vol = 0, 0, 0, 0
            for i in range(len(turning_points) - 1):
                window_df = minute_df.iloc[turning_points[i]:turning_points[i+1]+1]
                if window_df.empty: continue
                mf_net = window_df['main_force_net_vol'].sum()
                if window_df['minute_vwap'].iloc[-1] < window_df['minute_vwap'].iloc[0]: # 下跌波段
                    if mf_net > 0: suppressive_vol += mf_net
                    else: panic_vol += abs(mf_net)
                else: # 上涨波段
                    if mf_net < 0: rally_dist_vol += abs(mf_net)
                    else: rally_acc_vol += mf_net
            results['suppressive_accumulation_intensity'] = (suppressive_vol / total_daily_vol) * 100
            results['rally_distribution_intensity'] = (rally_dist_vol / total_daily_vol) * 100
            results['rally_accumulation_intensity'] = (rally_acc_vol / total_daily_vol) * 100
            results['panic_selling_intensity'] = (panic_vol / total_daily_vol) * 100
        # 2. 高级结构 (Advanced Structures)
        if pd.notna(close_price):
            high_20d, low_20d = context.get('high_20d'), context.get('low_20d')
            if pd.notna(high_20d) and pd.notna(low_20d):
                active_winners_df = self.df[(self.df['price'] >= low_20d) & (self.df['price'] <= high_20d) & (self.df['price'] < close_price)]
                if not active_winners_df.empty and active_winners_df['percent'].sum() > 0:
                    active_winner_avg_cost = np.average(active_winners_df['price'], weights=active_winners_df['percent'])
                    results['active_winner_avg_cost'] = active_winner_avg_cost
                    if active_winner_avg_cost > 0: results['active_winner_profit_margin'] = ((close_price - active_winner_avg_cost) / active_winner_avg_cost) * 100
            losers_df = self.df[self.df['price'] > close_price]
            if not losers_df.empty and losers_df['percent'].sum() > 0:
                results['loser_avg_cost'] = np.average(losers_df['price'], weights=losers_df['percent'])
        # 3. 获利盘信念 (Winner Conviction)
        active_profit_margin = results.get('active_winner_profit_margin')
        bullish_reinforcement = context.get('upward_impulse_purity')
        profit_taking_flow_ratio = context.get('profit_taking_flow_ratio')
        active_winner_rate = context.get('active_winner_rate')
        if all(pd.notna(v) for v in [active_profit_margin, bullish_reinforcement, profit_taking_flow_ratio, active_winner_rate]):
            realized_pressure = (profit_taking_flow_ratio / 100.0) / (active_winner_rate / 100.0) if active_winner_rate > 0 else 1.0
            hesitation_factor = 1.0 - np.clip(realized_pressure, 0, 1)
            margin_factor = np.log1p(np.clip(active_profit_margin / 100.0, 0, None))
            reinforcement_factor = 1.0 + (bullish_reinforcement / 100.0)
            results['winner_conviction_index'] = hesitation_factor * margin_factor * reinforcement_factor * 100
        # 4. 统一意图信号 (原 _calculate_intent_signals)
        results.update(self._calculate_intent_signals(context))
        return results
        # [代码新增结束]

    def _calculate_vital_sign_signals(self, context: dict) -> dict:
        """
        【V1.0 · 新增 · 生命体征信号旅】
        - 核心革命: 整合所有宏观的、总结性的元指标计算，形成统一的生命体征评估单元。
        - 包含逻辑: 成本共识与动量、结构韧性、主力姿态。
        """
        # [代码新增开始]
        results = {}
        # 1. 成本结构共识与筹码成本动量
        concentration_90pct = context.get('concentration_90pct')
        total_winner_rate = context.get('total_winner_rate')
        if all(pd.notna(v) for v in [concentration_90pct, total_winner_rate]):
            concentration_factor = 1.0 - np.clip(concentration_90pct, 0, 1)
            profit_factor = total_winner_rate / 100.0
            results['cost_structure_consensus_index'] = concentration_factor * profit_factor * 100
        short_term_cost = context.get('short_term_holder_cost')
        prev_short_term_cost = context.get('prev_short_term_holder_cost')
        prev_atr = context.get('prev_atr_14d')
        if all(pd.notna(v) for v in [short_term_cost, prev_short_term_cost, prev_atr]) and prev_atr > 0:
            results['chip_cost_momentum'] = (short_term_cost - prev_short_term_cost) / prev_atr
        # 2. 结构韧性指数
        def normalize(value, default=0.0): return value if pd.notna(value) else default
        consensus_index = normalize(results.get('cost_structure_consensus_index'))
        peak_profit_margin = normalize(context.get('dominant_peak_profit_margin'))
        foundation_score = np.log1p(consensus_index) * np.log1p(np.maximum(0, peak_profit_margin))
        active_winner_margin = normalize(context.get('active_winner_profit_margin'))
        winner_conviction = normalize(context.get('winner_conviction_index'))
        pressure_score = np.log1p(np.maximum(0, active_winner_margin)) * np.log1p(np.maximum(0, winner_conviction))
        cost_momentum = normalize(results.get('chip_cost_momentum'))
        upward_purity = normalize(context.get('upward_impulse_purity'))
        reinforcement_score = (np.tanh(cost_momentum) + 1) * ((upward_purity / 100.0) + 1)
        if foundation_score >= 0 and pressure_score >= 0 and reinforcement_score >= 0:
            resilience_raw = (foundation_score * pressure_score * reinforcement_score) ** (1.0 / 3.0)
            results['structural_resilience_index'] = np.clip(resilience_raw * 20, 0, 100)
        # 3. 主力姿态坐标
        resilience = normalize(results.get('structural_resilience_index')) / 100.0
        cost_advantage = np.tanh(normalize(context.get('main_force_cost_advantage')) / 10.0)
        loser_pressure = normalize(context.get('loser_capitulation_pressure_index'))
        pressure_penalty = np.tanh(loser_pressure / 50.0)
        results['posture_control_score'] = (resilience + cost_advantage) / 2.0 * (1 - pressure_penalty) * 100
        control_leverage = normalize(context.get('main_force_control_leverage')) / 100.0
        buy_intensity = normalize(context.get('suppressive_accumulation_intensity')) + normalize(context.get('rally_accumulation_intensity'))
        sell_intensity = normalize(context.get('rally_distribution_intensity')) + normalize(context.get('panic_selling_intensity'))
        tactical_factor = np.tanh((buy_intensity - sell_intensity) / 50.0)
        results['posture_action_score'] = (0.7 * control_leverage + 0.3 * tactical_factor) * 100
        return results
        # [代码新增结束]

    def _calculate_intent_signals(self, context: dict) -> dict:
        """
        【V1.0 · 新增 · 特种侦察连】
        - 核心革命: 整合“博弈意图”象限中多个单一职责的计算方法，形成统一的意图信号计算单元。
        - 包含逻辑: 主力成本优势、控盘杠杆、套牢盘压力、预估成本、新增套牢盘、收盘竞价、试探回升质量。
        """
        # [代码新增开始]
        import datetime
        results = {}
        minute_df = context.get('minute_data')
        # --- 1. 主力成本优势 (main_force_cost_advantage) ---
        required_cols_1 = ['minute_vwap', 'main_force_buy_vol', 'main_force_sell_vol', 'retail_buy_vol', 'retail_sell_vol']
        if minute_df is not None and not minute_df.empty and all(c in minute_df.columns for c in required_cols_1):
            mf_total_vol = minute_df['main_force_buy_vol'].sum() + minute_df['main_force_sell_vol'].sum()
            retail_total_vol = minute_df['retail_buy_vol'].sum() + minute_df['retail_sell_vol'].sum()
            if mf_total_vol > 0 and retail_total_vol > 0:
                mf_total_amount = (minute_df['main_force_buy_vol'] * minute_df['minute_vwap']).sum() + (minute_df['main_force_sell_vol'] * minute_df['minute_vwap']).sum()
                retail_total_amount = (minute_df['retail_buy_vol'] * minute_df['minute_vwap']).sum() + (minute_df['retail_sell_vol'] * minute_df['minute_vwap']).sum()
                vwap_mf = mf_total_amount / mf_total_vol
                vwap_retail = retail_total_amount / retail_total_vol
                if vwap_mf > 0: results['main_force_cost_advantage'] = (vwap_retail / vwap_mf - 1) * 100
        # --- 2. 主力控盘杠杆 (main_force_control_leverage) ---
        required_cols_2 = ['main_force_buy_vol', 'main_force_sell_vol']
        if minute_df is not None and not minute_df.empty and all(c in minute_df.columns for c in required_cols_2):
            mf_buy_vol = minute_df['main_force_buy_vol'].sum()
            mf_sell_vol = minute_df['main_force_sell_vol'].sum()
            if (mf_buy_vol + mf_sell_vol) > 0:
                results['main_force_control_leverage'] = ((mf_buy_vol - mf_sell_vol) / (mf_buy_vol + mf_sell_vol)) * 100
        # --- 3. 套牢盘投降压力 (loser_capitulation_pressure_index) ---
        loser_loss_margin = context.get('loser_loss_margin_avg')
        loser_concentration = context.get('loser_concentration_90pct')
        if all(pd.notna(v) for v in [loser_loss_margin, loser_concentration]):
            results['loser_capitulation_pressure_index'] = abs(loser_loss_margin) * loser_concentration
        # --- 4. 主力预估持仓成本 (estimated_main_force_position_cost) ---
        short_term_cost_line = context.get('short_term_holder_cost')
        if not self.df.empty and pd.notna(short_term_cost_line):
            foundational_chips_df = self.df[self.df['price'] < short_term_cost_line]
            if not foundational_chips_df.empty and foundational_chips_df['percent'].sum() > 0:
                results['estimated_main_force_position_cost'] = np.average(foundational_chips_df['price'], weights=foundational_chips_df['percent'])
        # --- 5. 日内新增套牢盘压力 (intraday_new_loser_pressure) ---
        close_price = context.get('close_price')
        daily_volume = context.get('daily_turnover_volume')
        if minute_df is not None and not minute_df.empty and pd.notna(close_price) and daily_volume and daily_volume > 0:
            new_losers_df = minute_df[minute_df['minute_vwap'] > close_price].copy()
            if not new_losers_df.empty:
                new_loser_vol = new_losers_df['vol_shares'].sum()
                if new_loser_vol > 0:
                    new_loser_vwap = (new_losers_df['minute_vwap'] * new_losers_df['vol_shares']).sum() / new_loser_vol
                    avg_loss_rate = abs((close_price / new_loser_vwap - 1))
                    results['intraday_new_loser_pressure'] = (new_loser_vol / daily_volume) * avg_loss_rate * 100
            else:
                results['intraday_new_loser_pressure'] = 0.0
        # --- 6. 集合竞价控盘信号 (closing_auction_control_signal) ---
        if minute_df is not None and not minute_df.empty and 'trade_time' in minute_df.columns:
            auction_df = minute_df[minute_df['trade_time'].dt.time >= datetime.time(14, 57)]
            if not auction_df.empty:
                total_auction_amount = (auction_df['vol_shares'] * auction_df['minute_vwap']).sum()
                if total_auction_amount > 0:
                    mf_net_amount = (auction_df['main_force_buy_vol'] * auction_df['minute_vwap']).sum() - (auction_df['main_force_sell_vol'] * auction_df['minute_vwap']).sum()
                    results['closing_auction_control_signal'] = (mf_net_amount / total_auction_amount) * 100
        # --- 7. 日内试探回升质量 (intraday_probe_rebound_quality) ---
        low_price = context.get('low_price')
        if minute_df is not None and not minute_df.empty and pd.notna(low_price):
            low_price_time = minute_df.loc[minute_df['low'].idxmin()]['trade_time'] if 'low' in minute_df.columns else minute_df.loc[minute_df['minute_vwap'].idxmin()]['trade_time']
            dip_df = minute_df[minute_df['trade_time'] <= low_price_time]
            rebound_df = minute_df[minute_df['trade_time'] > low_price_time]
            if not dip_df.empty and not rebound_df.empty:
                dip_total_volume = dip_df['vol_shares'].sum()
                if dip_total_volume > 0:
                    rebound_mf_net_vol = rebound_df['main_force_buy_vol'].sum() - rebound_df['main_force_sell_vol'].sum()
                    results['intraday_probe_rebound_quality'] = rebound_mf_net_vol / dip_total_volume
        return results
        # [代码新增结束]

    def _calculate_intraday_dynamics_signals(self, context: dict) -> dict:
        """
        【V1.0 · 新增 · 日内动态合成营】
        - 核心革命: 整合“内部动态”象限中多个零散计算方法，形成统一的日内动态计算单元。
        - 包含逻辑: 主峰动态、主动压力支撑、高级日内动态、战区交火剖析。
        """
        # [代码新增开始]
        results = {}
        minute_df = context.get('minute_data')
        total_daily_vol = context.get('daily_turnover_volume')
        # --- 1. 主峰动态 (Peak Dynamics) ---
        peak_range_low = context.get('peak_range_low')
        peak_range_high = context.get('peak_range_high')
        required_cols_1 = ['minute_vwap', 'vol', 'main_force_buy_vol', 'main_force_sell_vol', 'retail_buy_vol', 'retail_sell_vol']
        if minute_df is not None and not minute_df.empty and all(pd.notna(v) for v in [peak_range_low, peak_range_high, total_daily_vol]) and total_daily_vol > 0 and all(c in minute_df.columns for c in required_cols_1):
            peak_zone_mask = (minute_df['minute_vwap'] >= peak_range_low) & (minute_df['minute_vwap'] <= peak_range_high)
            peak_zone_df = minute_df[peak_zone_mask]
            if not peak_zone_df.empty:
                turnover_at_peak = peak_zone_df['vol'].sum()
                results['peak_battle_intensity'] = (turnover_at_peak / total_daily_vol) * 100
                if turnover_at_peak > 0:
                    mf_net_vol_peak = (peak_zone_df['main_force_buy_vol'] - peak_zone_df['main_force_sell_vol']).sum()
                    results['peak_mf_conviction_flow'] = (mf_net_vol_peak / turnover_at_peak) * 100
                mf_total_vol_peak = peak_zone_df['main_force_buy_vol'].sum() + peak_zone_df['main_force_sell_vol'].sum()
                retail_total_vol_peak = peak_zone_df['retail_buy_vol'].sum() + peak_zone_df['retail_sell_vol'].sum()
                if mf_total_vol_peak > 0 and retail_total_vol_peak > 0:
                    mf_amount_peak = (peak_zone_df['main_force_buy_vol'] * peak_zone_df['minute_vwap']).sum() + (peak_zone_df['main_force_sell_vol'] * peak_zone_df['minute_vwap']).sum()
                    retail_amount_peak = (peak_zone_df['retail_buy_vol'] * peak_zone_df['minute_vwap']).sum() + (peak_zone_df['retail_sell_vol'] * peak_zone_df['minute_vwap']).sum()
                    vwap_mf_peak = mf_amount_peak / mf_total_vol_peak
                    vwap_retail_peak = retail_amount_peak / retail_total_vol_peak
                    if vwap_retail_peak > 0: results['peak_main_force_premium'] = (vwap_mf_peak / vwap_retail_peak - 1) * 100
        # --- 2. 主动压力与支撑 (Active Pressure & Support) ---
        required_cols_2 = ['open', 'close', 'vol', 'main_force_buy_vol', 'main_force_sell_vol', 'retail_buy_vol', 'retail_sell_vol']
        if minute_df is not None and not minute_df.empty and all(c in minute_df.columns for c in required_cols_2):
            offensive_df = minute_df[minute_df['close'] > minute_df['open']]
            defensive_df = minute_df[minute_df['close'] < minute_df['open']]
            total_vol_on_rally = offensive_df['vol'].sum()
            if total_vol_on_rally > 0:
                total_sell_vol_on_rally = (offensive_df['main_force_sell_vol'] + offensive_df['retail_sell_vol']).sum()
                results['active_selling_pressure'] = (total_sell_vol_on_rally / total_vol_on_rally) * 100
            total_vol_on_dip = defensive_df['vol'].sum()
            if total_vol_on_dip > 0:
                total_buy_vol_on_dip = (defensive_df['main_force_buy_vol'] + defensive_df['retail_buy_vol']).sum()
                results['active_buying_support'] = (total_buy_vol_on_dip / total_vol_on_dip) * 100
        # --- 3. 高级日内动态 (Advanced Intraday Dynamics) ---
        required_cols_3 = ['open', 'close', 'vol', 'main_force_net_vol', 'high', 'low']
        if minute_df is not None and not minute_df.empty and all(c in minute_df.columns for c in required_cols_3):
            offensive_df = minute_df[minute_df['close'] > minute_df['open']]
            if not offensive_df.empty:
                total_vol_on_rally = offensive_df['vol'].sum()
                if total_vol_on_rally > 0:
                    mf_net_vol_on_rally = offensive_df['main_force_net_vol'].sum()
                    results['upward_impulse_purity'] = (mf_net_vol_on_rally / total_vol_on_rally) * 100
            pre_close = context.get('pre_close')
            open_price = context.get('open_price')
            if pd.notna(pre_close) and pre_close > 0 and pd.notna(open_price):
                gap = open_price - pre_close
                if abs(gap) > 1e-6:
                    opening_30min_df = minute_df[minute_df['trade_time'].dt.time < pd.to_datetime('10:00').time()]
                    if not opening_30min_df.empty:
                        gap_factor = gap / pre_close
                        fill_factor = ((open_price - opening_30min_df['low'].min()) / gap) if gap > 0 else ((opening_30min_df['high'].max() - open_price) / abs(gap))
                        total_vol_30min = opening_30min_df['vol'].sum()
                        force_factor = opening_30min_df['main_force_net_vol'].sum() / total_vol_30min if total_vol_30min > 0 else 0.0
                        results['opening_gap_defense_strength'] = gap_factor * (1 - min(1, max(0, fill_factor))) * force_factor * 10000
                    else: results['opening_gap_defense_strength'] = 0.0
                else: results['opening_gap_defense_strength'] = 0.0
        # --- 4. 战区交火剖析 (Combat Zone Dynamics) ---
        if minute_df is not None and not minute_df.empty and total_daily_vol and total_daily_vol > 0 and all(c in minute_df.columns for c in required_cols_1):
            high_20d = context.get('high_20d')
            low_20d = context.get('low_20d')
            if pd.notna(high_20d) and pd.notna(low_20d):
                active_zone_mask = (minute_df['minute_vwap'] >= low_20d) & (minute_df['minute_vwap'] <= high_20d)
                active_zone_minutes = minute_df[active_zone_mask]
                if not active_zone_minutes.empty:
                    vol_in_active_zone = active_zone_minutes['vol'].sum()
                    results['active_zone_combat_intensity'] = (vol_in_active_zone / total_daily_vol) * 100
                    if vol_in_active_zone > 0:
                        mf_net_vol_in_active_zone = (active_zone_minutes['main_force_buy_vol'] - active_zone_minutes['main_force_sell_vol']).sum()
                        results['active_zone_mf_stance'] = (mf_net_vol_in_active_zone / vol_in_active_zone) * 100
            prev_winner_avg_cost = context.get('prev_winner_avg_cost')
            if pd.notna(prev_winner_avg_cost) and prev_winner_avg_cost > 0:
                total_sell_vol = (minute_df['main_force_sell_vol'] + minute_df['retail_sell_vol']).sum()
                if total_sell_vol > 0:
                    total_sell_amount = ((minute_df['main_force_sell_vol'] + minute_df['retail_sell_vol']) * minute_df['minute_vwap']).sum()
                    results['profit_realization_quality'] = ((total_sell_amount / total_sell_vol) / prev_winner_avg_cost - 1) * 100
            prev_loser_avg_cost = context.get('prev_loser_avg_cost')
            if pd.notna(prev_loser_avg_cost) and prev_loser_avg_cost > 0:
                total_buy_vol = (minute_df['main_force_buy_vol'] + minute_df['retail_buy_vol']).sum()
                if total_buy_vol > 0:
                    total_buy_amount = ((minute_df['main_force_buy_vol'] + minute_df['retail_buy_vol']) * minute_df['minute_vwap']).sum()
                    results['capitulation_absorption_quality'] = ((total_buy_amount / total_buy_vol) / prev_loser_avg_cost - 1) * 100
        return results
        # [代码新增结束]

    def _calculate_cross_day_flow_signals(self, context: dict) -> dict:
        """
        【V1.0 · 新增 · 跨日流转合成营】
        - 核心革命: 整合“跨日迁徙”象限中多个零散计算方法，形成统一的跨日流转计算单元。
        - 包含逻辑: 控制权转移、流量质量与压力、成本分离度、情绪惯性、博弈疲劳度。
        """
        # [代码新增开始]
        results = {}
        minute_df = context.get('minute_data')
        total_daily_vol = context.get('daily_turnover_volume')
        # --- 1. 控制权净转移 (Concentration Dynamics) ---
        daily_vwap = context.get('daily_vwap')
        required_cols_1 = ['minute_vwap', 'main_force_buy_vol', 'main_force_sell_vol']
        if minute_df is not None and not minute_df.empty and all(pd.notna(v) for v in [daily_vwap, total_daily_vol]) and total_daily_vol > 0 and all(c in minute_df.columns for c in required_cols_1):
            minute_df['mf_net_vol'] = minute_df['main_force_buy_vol'] - minute_df['main_force_sell_vol']
            gathering_vol_per_minute = minute_df['mf_net_vol'].clip(lower=0)
            dispersal_vol_per_minute = (-minute_df['mf_net_vol']).clip(lower=0)
            below_vwap_mask = minute_df['minute_vwap'] < daily_vwap
            above_vwap_mask = minute_df['minute_vwap'] > daily_vwap
            results['gathering_by_support'] = (gathering_vol_per_minute[below_vwap_mask].sum() / total_daily_vol) * 100
            results['gathering_by_chasing'] = (gathering_vol_per_minute[above_vwap_mask].sum() / total_daily_vol) * 100
            results['dispersal_by_distribution'] = (dispersal_vol_per_minute[above_vwap_mask].sum() / total_daily_vol) * 100
            results['dispersal_by_capitulation'] = (dispersal_vol_per_minute[below_vwap_mask].sum() / total_daily_vol) * 100
        # --- 2. 流量质量与供给压力 (Flow Quality & Pressure) ---
        pre_close = context.get('pre_close')
        required_cols_2 = ['minute_vwap', 'main_force_sell_vol', 'retail_sell_vol']
        if minute_df is not None and not minute_df.empty and pd.notna(pre_close) and all(c in minute_df.columns for c in required_cols_2):
            total_sell_vol_today = (minute_df['main_force_sell_vol'] + minute_df['retail_sell_vol']).sum()
            if total_sell_vol_today > 0:
                profit_taking_vol = (minute_df[minute_df['minute_vwap'] > pre_close]['main_force_sell_vol'] + minute_df[minute_df['minute_vwap'] > pre_close]['retail_sell_vol']).sum()
                capitulation_vol = (minute_df[minute_df['minute_vwap'] < pre_close]['main_force_sell_vol'] + minute_df[minute_df['minute_vwap'] < pre_close]['retail_sell_vol']).sum()
                results['profit_taking_flow_ratio'] = (profit_taking_vol / total_sell_vol_today) * 100
                results['capitulation_flow_ratio'] = (capitulation_vol / total_sell_vol_today) * 100
        prev_df = context.get('prev_chip_distribution')
        prev_high_20d, prev_low_20d, prev_total_chips = context.get('prev_high_20d'), context.get('prev_low_20d'), context.get('prev_total_chip_volume')
        if prev_df is not None and not prev_df.empty and all(pd.notna(v) for v in [prev_high_20d, prev_low_20d, prev_total_chips]) and prev_total_chips > 0 and total_sell_vol_today > 0:
            prev_active_mask = (prev_df['price'] >= prev_low_20d) & (prev_df['price'] <= prev_high_20d)
            vol_active_winners = (prev_df[prev_active_mask & (prev_df['price'] < pre_close)]['percent'].sum() / 100) * prev_total_chips
            vol_locked_winners = (prev_df[~prev_active_mask & (prev_df['price'] < pre_close)]['percent'].sum() / 100) * prev_total_chips
            vol_active_losers = (prev_df[prev_active_mask & (prev_df['price'] > pre_close)]['percent'].sum() / 100) * prev_total_chips
            vol_locked_losers = (prev_df[~prev_active_mask & (prev_df['price'] > pre_close)]['percent'].sum() / 100) * prev_total_chips
            if vol_active_winners > 0: results['active_winner_pressure_ratio'] = (total_sell_vol_today / vol_active_winners) * 100
            if vol_locked_winners > 0: results['locked_profit_pressure_ratio'] = (total_sell_vol_today / vol_locked_winners) * 100
            if vol_active_losers > 0: results['active_loser_pressure_ratio'] = (total_sell_vol_today / vol_active_losers) * 100
            if vol_locked_losers > 0: results['locked_loss_pressure_ratio'] = (total_sell_vol_today / vol_locked_losers) * 100
        # --- 3. 成本分离度 (Cost Divergence) ---
        high_20d, low_20d = context.get('high_20d'), context.get('low_20d')
        if not self.df.empty and all(pd.notna(v) for v in [high_20d, low_20d]):
            active_chips_df = self.df[(self.df['price'] >= low_20d) & (self.df['price'] <= high_20d)]
            foundational_chips_df = self.df[self.df['price'] < low_20d]
            avg_cost_active = np.average(active_chips_df['price'], weights=active_chips_df['percent']) if not active_chips_df.empty and active_chips_df['percent'].sum() > 0 else None
            avg_cost_foundational = np.average(foundational_chips_df['price'], weights=foundational_chips_df['percent']) if not foundational_chips_df.empty and foundational_chips_df['percent'].sum() > 0 else None
            if avg_cost_active is not None and avg_cost_foundational is not None and avg_cost_foundational > 0:
                divergence = (avg_cost_active / avg_cost_foundational - 1) * 100
                results['cost_divergence'] = divergence
                atr_14d, close_price = context.get('atr_14d'), context.get('close_price')
                if pd.notna(atr_14d) and atr_14d > 0 and pd.notna(close_price) and close_price > 0:
                    volatility_pct = (atr_14d / close_price) * 100
                    if volatility_pct > 0: results['cost_divergence_normalized'] = divergence / volatility_pct
        # --- 4. 情绪惯性 (Emotional Inertia) ---
        close_price, total_chip_volume = context.get('close_price'), context.get('total_chip_volume')
        if not self.df.empty and all(pd.notna(v) for v in [close_price, pre_close, total_chip_volume, total_daily_vol]) and total_chip_volume > 0 and total_daily_vol > 0:
            critical_chips_df = self.df[(self.df['price'] > min(close_price, pre_close)) & (self.df['price'] < max(close_price, pre_close))]
            status_change_net_flow_pct = critical_chips_df['percent'].sum() * np.sign(close_price - pre_close)
            turnover_rate = total_daily_vol / total_chip_volume
            if turnover_rate > 0: results['winner_loser_momentum'] = status_change_net_flow_pct / turnover_rate
        # --- 5. 博弈疲劳度 (Game Fatigue) ---
        prev_fatigue_index, recent_closes = context.get('prev_chip_fatigue_index', 0.0), context.get('recent_10d_closes')
        if all(pd.notna(v) for v in [prev_fatigue_index, close_price, total_chip_volume, total_daily_vol]) and recent_closes is not None and len(recent_closes) >= 9:
            is_effective_day = (close_price >= max(recent_closes)) or (close_price <= min(recent_closes))
            fatigue_index = prev_fatigue_index * 0.98
            if is_effective_day: fatigue_index *= 0.5
            else: fatigue_index += (total_daily_vol / total_chip_volume) * 100 if total_chip_volume > 0 else 0
            results['chip_fatigue_index'] = fatigue_index
        return results
        # [代码新增结束]

    def _calculate_chip_structure_health_score(self, context: dict) -> dict:
        """
        【V3.0 · 动态百分位归一化版】计算筹码结构健康分。
        - 核心革命: 引入基于真实历史数据的动态百分位排名进行归一化，实现模型自适应。
        - 核心架构: 沿用“结构稳固度”、“动能纯粹度”、“内部压力度”三维几何平均模型。
        """
        from scipy.stats import percentileofscore
        results = {'chip_health_score': None}
        # 1. 定义三维模型的组件和权重(方向)
        model_dimensions = {
            'structural_soundness': {
                'components': {
                    'concentration_70pct': -1, # 越低越好
                    'cost_divergence_normalized': -1, # 越接近0或负值越好
                    'dominant_peak_profit_margin': 1, # 越高越好
                }
            },
            'momentum_purity': {
                'components': {
                    'main_force_cost_advantage': 1, # 越高越好
                    'suppressive_accumulation_intensity': 1, # 越高越好
                    'upward_impulse_purity': 1, # 越高越好
                }
            },
            'internal_pressure': {
                'components': {
                    'active_winner_profit_margin': 1, # 越高越好
                    'winner_conviction_index': 1, # 越高越好
                }
            }
        }
        # 2. 获取历史数据
        historical_df = context.get('historical_components')
        if historical_df is None or historical_df.empty:
            logger.debug(f"[{context.get('stock_code')}] [{context.get('trade_date')}] 健康分计算跳过，因缺少历史数据进行动态归一化。")
            return results
        dimension_scores = {}
        # 3. 逐个维度计算得分
        for dim_name, dim_data in model_dimensions.items():
            component_scores = []
            for metric, weight in dim_data['components'].items():
                current_value = context.get(metric)
                if current_value is None or not pd.notna(current_value) or metric not in historical_df.columns:
                    continue # 如果当前值或历史数据缺失，则跳过
                # 获取该指标的历史序列
                historical_series = historical_df[metric].dropna()
                if historical_series.empty:
                    continue
                # 计算当前值在历史数据中的百分位排名
                percentile = percentileofscore(historical_series, current_value, kind='rank') / 100.0
                # 根据权重调整归一化分数
                normalized_score = percentile if weight > 0 else (1.0 - percentile)
                component_scores.append(normalized_score)
            # 如果该维度有有效分数，则计算其平均分
            if component_scores:
                dimension_scores[dim_name] = np.mean(component_scores)
        # 4. 使用几何平均整合各维度得分
        if not dimension_scores or len(dimension_scores) < len(model_dimensions):
            return results # 如果任何一个维度无法计算，则整体健康分无效
        final_score_raw = 1.0
        for score in dimension_scores.values():
            final_score_raw *= score
        # 开立方根（或N次方根）
        final_score_normalized = final_score_raw ** (1.0 / len(dimension_scores))
        # 映射到 0-100 分
        results['chip_health_score'] = final_score_normalized * 100
        return results

    def _calculate_profit_loss_quality(self, chip_df: pd.DataFrame, close_price: float) -> dict:
        """
        【V1.0 · 新增 · 盈利亏损质量模型】
        - 核心思想: 量化“盈利的厚度”与“亏损的痛感”，超越简单的盈亏比例。
        - 新增指标:
          - winner_profit_cushion: 获利盘缓冲垫。衡量获利盘能承受多大的价格回撤而不亏损。
          - loser_pain_index: 套牢盘痛苦指数。结合了亏损深度和套牢盘的集中度。
        """
        # [代码新增开始]
        results = {
            'winner_profit_cushion': None,
            'loser_pain_index': None,
        }
        if chip_df.empty or not pd.notna(close_price):
            return results
        # --- 1. 计算获利盘缓冲垫 (Winner Profit Cushion) ---
        winners_df = chip_df[chip_df['price'] < close_price].copy()
        if not winners_df.empty and winners_df['percent'].sum() > 0:
            # 归一化获利盘内部的筹码分布
            winners_df['percent'] = (winners_df['percent'] / winners_df['percent'].sum()) * 100
            winners_df['cum_percent'] = winners_df['percent'].cumsum()
            # 找到成本最高的15%获利盘的成本线 (85%分位点)
            # 这是最容易叛变的获利盘，他们的成本线是多方的第一道心理防线
            cost_at_85pct = np.interp(85, winners_df['cum_percent'], winners_df['price'])
            if pd.notna(cost_at_85pct) and cost_at_85pct > 0:
                # 缓冲垫 = (收盘价 / 这部分最不坚定的获利盘成本 - 1) * 100
                results['winner_profit_cushion'] = (close_price / cost_at_85pct - 1) * 100
        # --- 2. 计算套牢盘痛苦指数 (Loser Pain Index) ---
        losers_df = chip_df[chip_df['price'] > close_price].copy()
        if not losers_df.empty and losers_df['percent'].sum() > 0:
            # 计算套牢盘的平均亏损幅度
            loser_avg_cost = np.average(losers_df['price'], weights=losers_df['percent'])
            avg_loss_margin = abs((close_price / loser_avg_cost - 1) * 100) if loser_avg_cost > 0 else 0
            # 计算套牢盘的内部集中度 (90%筹码的成本区间宽度 / 平均成本)
            losers_df['percent'] = (losers_df['percent'] / losers_df['percent'].sum()) * 100
            losers_df['cum_percent'] = losers_df['percent'].cumsum()
            cost_5pct = np.interp(5, losers_df['cum_percent'], losers_df['price'])
            cost_95pct = np.interp(95, losers_df['cum_percent'], losers_df['price'])
            loser_concentration = (cost_95pct - cost_5pct) / loser_avg_cost if loser_avg_cost > 0 else 0
            # 痛苦指数 = 平均亏损幅度 * (1 - 集中度)
            # 集中度越低（越分散），越不容易形成合力，痛苦感越强；反之，成本接近，容易抱团。
            results['loser_pain_index'] = avg_loss_margin * (1 - np.clip(loser_concentration, 0, 1))
        return results
        # [代码新增结束]

    def _calculate_structural_stability(self, context: dict) -> dict:
        """
        【V1.0 · 新增 · 结构稳定性评估模型】
        - 核心思想: 将多个关键静态指标融合成一个综合评分，量化当前筹码结构的“稳固程度”。
        - 新增指标:
          - structural_stability_score: 结构稳定性评分 (0-100)。
        """
        # [代码新增开始]
        results = {'structural_stability_score': None}
        # 1. 获取计算所需的组件指标
        # 集中度因子: 70%筹码集中度，越低越好
        concentration_70pct = self.ctx.get('concentration_70pct')
        # 盈利因子: 总获利盘比例，越高越好
        total_winner_rate = context.get('total_winner_rate')
        # 缓冲垫因子: 获利盘缓冲垫，越高越好
        winner_profit_cushion = context.get('winner_profit_cushion')
        # 峰位因子: 主峰利润边际，越高越好
        dominant_peak_profit_margin = context.get('dominant_peak_profit_margin')
        # 检查所有组件是否有效
        if not all(pd.notna(v) for v in [concentration_70pct, total_winner_rate, winner_profit_cushion, dominant_peak_profit_margin]):
            return results
        # 2. 将各因子归一化到 [0, 1] 区间
        # 集中度得分: 使用exp(-x)函数，集中度越低(接近0)，得分越高(接近1)
        concentration_score = np.exp(-5 * np.clip(concentration_70pct, 0, None))
        # 盈利比例得分
        winner_rate_score = total_winner_rate / 100.0
        # 缓冲垫得分: 使用tanh函数进行S型归一化，超过20%的缓冲垫效果提升有限
        cushion_score = np.tanh(winner_profit_cushion / 10.0)
        # 主峰利润得分: 同样使用tanh归一化
        peak_profit_score = np.tanh(dominant_peak_profit_margin / 20.0)
        # 3. 使用几何平均法整合各维度得分，任何一个短板都会显著拉低总分
        # 确保所有分数为正
        scores = [concentration_score, winner_rate_score, cushion_score, peak_profit_score]
        valid_scores = [s for s in scores if s > 0]
        if not valid_scores:
            return results
        stability_raw = 1.0
        for score in valid_scores:
            stability_raw *= score
        final_score = stability_raw ** (1.0 / len(valid_scores))
        results['structural_stability_score'] = final_score * 100
        return results
        # [代码新增结束]

    def _calculate_cost_distribution_statistics(self, chip_df: pd.DataFrame) -> dict:
        """
        【V1.0 · 新增 · 成本分布形态分析模型】
        - 核心思想: 利用统计学方法计算筹码分布的偏度，判断成本重心的偏移方向。
        - 新增指标:
          - cost_structure_skewness: 成本结构偏度。
        """
        # [代码新增开始]
        results = {'cost_structure_skewness': None}
        if chip_df.empty or chip_df['percent'].sum() < 1e-6:
            return results
        from scipy.stats import skew
        # 为了计算加权偏度，我们需要“解构”加权分布
        # 将百分比转换为整数权重，以避免浮点数问题并提高效率
        # 将总权重缩放到一个合理的大小（如10000），以避免创建过大的数组
        total_percent = chip_df['percent'].sum()
        weights = np.round((chip_df['percent'] / total_percent) * 10000).astype(int)
        # 过滤掉权重为0的行
        valid_weights = weights[weights > 0]
        if len(valid_weights) < 3: # 偏度计算至少需要3个点
            return results
        valid_prices = chip_df['price'][weights > 0]
        # 根据权重重复价格数据，构建一个等效的非加权样本分布
        unweighted_sample = np.repeat(valid_prices, valid_weights)
        # 计算偏度
        # 左偏(负值): 分布的尾部在左侧，大部分成本集中在右侧(高价区)，看跌。
        # 右偏(正值): 分布的尾部在右侧，大部分成本集中在左侧(低价区)，看涨。
        # 注意：scipy.stats.skew 的定义与常规金融理解相反，我们取其负值以符合直觉。
        # 金融直觉：成本重心偏向低价区（右偏分布）是好事，应为正。
        skewness = skew(unweighted_sample)
        results['cost_structure_skewness'] = -skewness
        return results
        # [代码新增结束]












