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
        """
        【V12.1 · 逐笔数据兼容版】
        - 核心重构: 建立“数据净化协议”，在入口处将所有可能为Decimal的上下文数据强制转换为float，根除类型不匹配错误。
        - 核心新增: 引入 `intraday_data` 作为统一的日内数据源，优先使用逐笔数据，否则回退到分钟数据。
        """
        self.df = daily_chips_df.reset_index(drop=True)
        self.ctx = context_data
        if not self.df.empty:
            percent_sum = self.df['percent'].sum()
            if not np.isclose(percent_sum, 100.0) and percent_sum > 0:
                self.df['percent'] = (self.df['percent'] / percent_sum) * 100.0
        decimal_to_float_fields = [
            'total_chip_volume', 'daily_turnover_volume', 'close_price', 'high_price', 'low_price', 'open_price',
            'pre_close', 'prev_20d_close', 'circ_mv', 'cost_5pct', 'cost_15pct', 'cost_50pct', 'cost_85pct',
            'cost_95pct', 'weight_avg', 'winner_rate', 'prev_concentration_90pct', 'prev_winner_avg_cost',
            'prev_total_chip_volume',
            'prev_high_20d', 'prev_low_20d', 'prev_day_20d_ago_close'
        ]
        for key in decimal_to_float_fields:
            if key in self.ctx and isinstance(self.ctx[key], Decimal):
                self.ctx[key] = float(self.ctx[key])
        self._prepare_intraday_data_features()

    def calculate_all_metrics(self) -> dict:
        """
        【V15.1 · 微观博弈整合版】
        - 核心新增: 增加第六个计算单元 `_compute_microstructure_game_metrics`，专门负责计算基于高频数据的筹码微观博弈指标。
        - 核心重构: 废除中间调度方法，由本方法直接指挥六大计算单元，实现扁平化、高效的指挥链。
        - 核心思想: 将所有计算逻辑整合为六个高度内聚的“合成营”方法，并按依赖顺序依次执行。
        """
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
        # 步骤1: 提取基础指标并更新上下文
        summary_info = self._get_summary_metrics_from_context()
        self.ctx.update(summary_info)
        # 步骤2: 按依赖顺序调用六大计算单元，并持续更新上下文
        static_structure_metrics = self._compute_static_structure_metrics()
        self.ctx.update(static_structure_metrics)
        intraday_dynamics_metrics = self._compute_intraday_dynamics_metrics(self.ctx)
        self.ctx.update(intraday_dynamics_metrics)
        cross_day_flow_metrics = self._compute_cross_day_flow_metrics(self.ctx)
        self.ctx.update(cross_day_flow_metrics)
        game_theoretic_metrics = self._compute_game_theoretic_metrics(self.ctx)
        self.ctx.update(game_theoretic_metrics)
        vital_signs_metrics = self._compute_vital_sign_metrics(self.ctx)
        self.ctx.update(vital_signs_metrics)
        microstructure_game_metrics = self._compute_microstructure_game_metrics(self.ctx)
        self.ctx.update(microstructure_game_metrics)
        # 步骤3: 整合所有计算结果
        all_metrics = {
            **summary_info,
            **static_structure_metrics,
            **intraday_dynamics_metrics,
            **cross_day_flow_metrics,
            **game_theoretic_metrics,
            **vital_signs_metrics,
            **microstructure_game_metrics,
        }
        # 步骤4: 计算最终的健康分 (它依赖于之前计算出的所有指标)
        health_score_info = self._calculate_chip_structure_health_score(self.ctx)
        all_metrics.update(health_score_info)
        # 步骤5: 清理临时或中间指标
        all_metrics.pop('peak_range_low', None)
        all_metrics.pop('peak_range_high', None)
        return all_metrics

    def _prepare_intraday_data_features(self):
        """
        【V2.10 · 日内数据列顺序修复版】准备日内数据特征，统一处理为分钟级别数据。
        - 核心进化: 能够直接处理由上游服务传入的、已经聚合和归因完毕的分钟数据。
        - 核心逻辑: 假设传入的 `intraday_data` 已经是分钟级别数据，并进行标准化处理。
        - 核心修复: 调整 `amount_yuan` 和 `vol_shares` 的创建顺序，确保在计算 `minute_vwap` 前它们已存在。
        """
        intraday_df = self.ctx.get('intraday_data')
        stock_code = self.ctx.get('stock_code', 'UNKNOWN')
        trade_date = self.ctx.get('trade_date', 'UNKNOWN')
        if intraday_df is None or intraday_df.empty:
            logger.warning(f"[{stock_code}] [{trade_date}] 日内数据特征准备跳过，原因：日内数据(intraday_data)为空。")
            self.ctx.update({'daily_vwap': None, 'volume_above_vwap_ratio': None, 'volume_below_vwap_ratio': None, 'processed_intraday_df': pd.DataFrame()})
            return
        processed_intraday_df = intraday_df.copy()
        # 1. 转换核心列为数值类型
        dtype_map = {'amount': 'float32', 'vol': 'int32', 'open': 'float32', 'close': 'float32', 'high': 'float32', 'low': 'float32'}
        for col, dtype in dtype_map.items():
            if col in processed_intraday_df.columns:
                processed_intraday_df[col] = pd.to_numeric(processed_intraday_df[col], errors='coerce').astype(dtype, errors='ignore')
        # 2. 确保 'amount_yuan' 和 'vol_shares' 存在
        # 修改行: 确保 vol_shares 在 minute_vwap 之前创建
        if 'vol_shares' not in processed_intraday_df.columns:
            if 'vol' in processed_intraday_df.columns:
                processed_intraday_df['vol_shares'] = processed_intraday_df['vol']
            else:
                processed_intraday_df['vol_shares'] = 0.0 # 如果原始 'vol' 也缺失，则默认为0
        # 修改行: 确保 amount_yuan 在 minute_vwap 之前创建
        if 'amount_yuan' not in processed_intraday_df.columns:
            if 'amount' in processed_intraday_df.columns:
                processed_intraday_df['amount_yuan'] = processed_intraday_df['amount']
            else:
                processed_intraday_df['amount_yuan'] = 0.0 # 如果原始 'amount' 也缺失，则默认为0
        # 3. 计算 'minute_vwap' (现在 'amount_yuan' 和 'vol_shares' 保证存在)
        if 'minute_vwap' not in processed_intraday_df.columns:
            processed_intraday_df['minute_vwap'] = processed_intraday_df['amount_yuan'] / processed_intraday_df['vol_shares'].replace(0, np.nan)
        # 4. 确保资金流相关列存在
        fund_flow_cols = ['main_force_net_vol', 'main_force_buy_vol', 'main_force_sell_vol', 'retail_buy_vol', 'retail_sell_vol', 'buy_vol_raw', 'sell_vol_raw']
        for col in fund_flow_cols:
            if col not in processed_intraday_df.columns:
                processed_intraday_df[col] = 0.0
        if processed_intraday_df.empty:
            self.ctx.update({'daily_vwap': None, 'volume_above_vwap_ratio': None, 'volume_below_vwap_ratio': None, 'processed_intraday_df': pd.DataFrame()})
            return
        total_amount_yuan = processed_intraday_df['amount_yuan'].sum()
        total_vol_shares = processed_intraday_df['vol_shares'].sum()
        daily_vwap = total_amount_yuan / total_vol_shares if total_vol_shares > 0 else None
        self.ctx['daily_vwap'] = daily_vwap
        if daily_vwap is None:
            self.ctx.update({'volume_above_vwap_ratio': None, 'volume_below_vwap_ratio': None, 'processed_intraday_df': processed_intraday_df})
            return
        vol_above_vwap = processed_intraday_df[processed_intraday_df['minute_vwap'] > daily_vwap]['vol_shares'].sum()
        vol_below_vwap = processed_intraday_df[processed_intraday_df['minute_vwap'] < daily_vwap]['vol_shares'].sum()
        total_vol = processed_intraday_df['vol_shares'].sum()
        if total_vol > 0:
            self.ctx['volume_above_vwap_ratio'] = vol_above_vwap / total_vol
            self.ctx['volume_below_vwap_ratio'] = vol_below_vwap / total_vol
        else:
            self.ctx.update({'volume_above_vwap_ratio': 0.0, 'volume_below_vwap_ratio': 0.0})
        self.ctx['processed_intraday_df'] = processed_intraday_df

    def _get_summary_metrics_from_context(self) -> dict:
        """
        【V14.0 · 稳定性前置注入版】
        - 核心新增: 提取并计算 `concentration_70pct`，为 `structural_stability_score` 提供关键前置条件。
        """
        weight_avg_cost = self.ctx.get('weight_avg')
        total_winner_rate = self.ctx.get('winner_rate')
        # 为 structural_stability_score 计算提供前置条件
        cost_15pct = self.ctx.get('cost_15pct')
        cost_85pct = self.ctx.get('cost_85pct')
        concentration_70pct = None
        if all(pd.notna(v) for v in [cost_15pct, cost_85pct, weight_avg_cost]) and weight_avg_cost > 0:
            concentration_70pct = (cost_85pct - cost_15pct) / weight_avg_cost
        return {
            'weight_avg_cost': weight_avg_cost,
            'total_winner_rate': total_winner_rate,
            'concentration_70pct': concentration_70pct, # 注入上下文
        }

    def _compute_cross_day_flow_metrics(self, context: dict) -> dict:
        """
        【V2.3 · 逐笔数据兼容版】
        - 核心修复: 初始化 `total_sell_vol_today` 局部变量，避免 `UnboundLocalError`。
        - 核心新增: 使用 `processed_intraday_df` 作为日内数据源。
        """
        results = {}
        intraday_df = context.get('processed_intraday_df')
        total_daily_vol = context.get('daily_turnover_volume')
        daily_vwap = context.get('daily_vwap')
        required_cols_1 = ['minute_vwap', 'main_force_buy_vol', 'main_force_sell_vol']
        if intraday_df is not None and not intraday_df.empty and all(pd.notna(v) for v in [daily_vwap, total_daily_vol]) and total_daily_vol > 0 and all(c in intraday_df.columns for c in required_cols_1):
            intraday_df['mf_net_vol'] = intraday_df['main_force_buy_vol'] - intraday_df['main_force_sell_vol']
            gathering_vol_per_minute = intraday_df['mf_net_vol'].clip(lower=0)
            dispersal_vol_per_minute = (-intraday_df['mf_net_vol']).clip(lower=0)
            below_vwap_mask = intraday_df['minute_vwap'] < daily_vwap
            above_vwap_mask = intraday_df['minute_vwap'] > daily_vwap
            results['gathering_by_support'] = (gathering_vol_per_minute[below_vwap_mask].sum() / total_daily_vol) * 100
            results['gathering_by_chasing'] = (gathering_vol_per_minute[above_vwap_mask].sum() / total_daily_vol) * 100
            results['dispersal_by_distribution'] = (dispersal_vol_per_minute[above_vwap_mask].sum() / total_daily_vol) * 100
            results['dispersal_by_capitulation'] = (dispersal_vol_per_minute[below_vwap_mask].sum() / total_daily_vol) * 100
        pre_close = context.get('pre_close')
        required_cols_2 = ['minute_vwap', 'main_force_sell_vol', 'retail_sell_vol']
        total_sell_vol_today = 0.0
        if intraday_df is not None and not intraday_df.empty and pd.notna(pre_close) and all(c in intraday_df.columns for c in required_cols_2):
            total_sell_vol_today = (intraday_df['main_force_sell_vol'] + intraday_df['retail_sell_vol']).sum()
            if total_sell_vol_today > 0:
                profit_taking_vol = (intraday_df[intraday_df['minute_vwap'] > pre_close]['main_force_sell_vol'] + intraday_df[intraday_df['minute_vwap'] > pre_close]['retail_sell_vol']).sum()
                capitulation_vol = (intraday_df[intraday_df['minute_vwap'] < pre_close]['main_force_sell_vol'] + intraday_df[intraday_df['minute_vwap'] < pre_close]['retail_sell_vol']).sum()
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
        close_price, total_chip_volume = context.get('close_price'), context.get('total_chip_volume')
        if not self.df.empty and all(pd.notna(v) for v in [close_price, pre_close, total_chip_volume, total_daily_vol]) and total_chip_volume > 0 and total_daily_vol > 0:
            critical_chips_df = self.df[(self.df['price'] > min(close_price, pre_close)) & (self.df['price'] < max(close_price, pre_close))]
            status_change_net_flow_pct = critical_chips_df['percent'].sum() * np.sign(close_price - pre_close)
            turnover_rate = total_daily_vol / total_chip_volume
            if turnover_rate > 0: results['winner_loser_momentum'] = status_change_net_flow_pct / turnover_rate
        prev_fatigue_index, recent_closes = context.get('prev_chip_fatigue_index', 0.0), context.get('recent_10d_closes')
        if all(pd.notna(v) for v in [prev_fatigue_index, close_price, total_chip_volume, total_daily_vol]) and recent_closes is not None and len(recent_closes) >= 9:
            is_effective_day = (close_price >= max(recent_closes)) or (close_price <= min(recent_closes))
            fatigue_index = prev_fatigue_index * 0.98
            if is_effective_day: fatigue_index *= 0.5
            else: fatigue_index += (total_daily_vol / total_chip_volume) * 100 if total_chip_volume > 0 else 0
            results['chip_fatigue_index'] = fatigue_index
        prev_chip_dist = context.get('prev_chip_distribution')
        prev_dominant_peak_cost = context.get('prev_dominant_peak_cost')
        today_dominant_peak_cost = context.get('dominant_peak_cost')
        if prev_chip_dist is not None and not prev_chip_dist.empty and all(pd.notna(v) for v in [prev_dominant_peak_cost, today_dominant_peak_cost]):
            prev_upper_shoulder_df = prev_chip_dist[(prev_chip_dist['price'] >= prev_dominant_peak_cost * 1.02) & (prev_chip_dist['price'] <= prev_dominant_peak_cost * 1.07)]
            prev_lower_shoulder_df = prev_chip_dist[(prev_chip_dist['price'] >= prev_dominant_peak_cost * 0.93) & (prev_chip_dist['price'] <= prev_dominant_peak_cost * 0.98)]
            today_upper_shoulder_df = self.df[(self.df['price'] >= today_dominant_peak_cost * 1.02) & (self.df['price'] <= today_dominant_peak_cost * 1.07)]
            today_lower_shoulder_df = self.df[(self.df['price'] >= today_dominant_peak_cost * 0.93) & (self.df['price'] <= today_dominant_peak_cost * 0.98)]
            chip_vol_upper_yesterday = prev_upper_shoulder_df['percent'].sum()
            chip_vol_lower_yesterday = prev_lower_shoulder_df['percent'].sum()
            chip_vol_upper_today = today_upper_shoulder_df['percent'].sum()
            chip_vol_lower_today = today_lower_shoulder_df['percent'].sum()
            upper_growth = (chip_vol_upper_today / chip_vol_upper_yesterday - 1) * 100 if chip_vol_upper_yesterday > 0 else (100 if chip_vol_upper_today > 0 else 0)
            lower_growth = (chip_vol_lower_today / chip_vol_lower_yesterday - 1) * 100 if chip_vol_lower_yesterday > 0 else (100 if chip_vol_lower_today > 0 else 0)
            results['peak_shoulder_growth_rate'] = upper_growth - lower_growth
        return results

    def _compute_game_theoretic_metrics(self, context: dict) -> dict:
        """
        【V7.9 · 逐笔数据兼容版】移除所有诊断探针。
        - 核心新增: 使用 `processed_intraday_df` 作为日内数据源。
        - 【修复】重新定义 `active_winner_profit_margin` 的计算逻辑，使其在价格突破近期高点时仍能有效捕捉活跃获利盘的利润垫。
        - 【修正】优化 `winner_conviction_index` 的计算，确保在涨停日等积极行情下能正确反映赢家信念。
        - 【修正】将 `active_profit_margin` 作为参数直接传递给 `_calculate_winner_conviction_index`，解决 `NoneType` 错误。
        """
        import datetime
        results = {}
        intraday_df = context.get('processed_intraday_df')
        total_daily_vol = context.get('daily_turnover_volume')
        close_price = context.get('close_price')
        atr_14d = context.get('atr_14d')
        required_cols_1 = ['minute_vwap', 'main_force_buy_vol', 'main_force_sell_vol']
        if intraday_df is not None and not intraday_df.empty and pd.notna(total_daily_vol) and total_daily_vol > 0 and all(c in intraday_df.columns for c in required_cols_1):
            peaks, _ = find_peaks(intraday_df['minute_vwap'], prominence=0.001)
            troughs, _ = find_peaks(-intraday_df['minute_vwap'], prominence=0.001)
            turning_points = sorted(list(set(np.concatenate(([0], troughs, peaks, [len(intraday_df)-1])))))
            suppressive_vol, rally_dist_vol, rally_acc_vol, panic_vol = 0, 0, 0, 0
            for i in range(len(turning_points) - 1):
                window_df = intraday_df.iloc[turning_points[i]:turning_points[i+1]+1]
                if window_df.empty: continue
                mf_net = window_df['main_force_net_vol'].sum()
                if window_df['minute_vwap'].iloc[-1] < window_df['minute_vwap'].iloc[0]:
                    if mf_net > 0: suppressive_vol += mf_net
                    else: panic_vol += abs(mf_net)
                else:
                    if mf_net < 0: rally_dist_vol += abs(mf_net)
                    else: rally_acc_vol += mf_net
            results['suppressive_accumulation_intensity'] = (suppressive_vol / total_daily_vol) * 100
            results['rally_distribution_intensity'] = (rally_dist_vol / total_daily_vol) * 100
            results['rally_accumulation_intensity'] = (rally_acc_vol / total_daily_vol) * 100
            results['panic_selling_intensity'] = (panic_vol / total_daily_vol) * 100
        active_winner_avg_cost, active_profit_margin = self._calculate_active_winner_profit_margin(close_price, atr_14d, context)
        results['active_winner_avg_cost'] = active_winner_avg_cost
        results['active_winner_profit_margin'] = active_profit_margin
        losers_df = self.df[self.df['price'] > close_price]
        if not losers_df.empty and losers_df['percent'].sum() > 0:
            results['loser_avg_cost'] = np.average(losers_df['price'], weights=losers_df['percent'])
        results['winner_conviction_index'] = self._calculate_winner_conviction_index(context, active_profit_margin)
        required_cols_4_1 = ['minute_vwap', 'main_force_buy_vol', 'main_force_sell_vol', 'retail_buy_vol', 'retail_sell_vol']
        if intraday_df is not None and not intraday_df.empty and all(c in intraday_df.columns for c in required_cols_4_1):
            mf_total_vol = intraday_df['main_force_buy_vol'].sum() + intraday_df['main_force_sell_vol'].sum()
            retail_total_vol = intraday_df['retail_buy_vol'].sum() + intraday_df['retail_sell_vol'].sum()
            if mf_total_vol > 0 and retail_total_vol > 0:
                mf_total_amount = (intraday_df['main_force_buy_vol'] * intraday_df['minute_vwap']).sum() + (intraday_df['main_force_sell_vol'] * intraday_df['minute_vwap']).sum()
                retail_total_amount = (intraday_df['retail_buy_vol'] * intraday_df['minute_vwap']).sum() + (intraday_df['retail_sell_vol'] * intraday_df['minute_vwap']).sum()
                vwap_mf = mf_total_amount / mf_total_vol
                vwap_retail = retail_total_amount / retail_total_vol
                if vwap_mf > 0: results['main_force_cost_advantage'] = (vwap_retail / vwap_mf - 1) * 100
        required_cols_4_2 = ['main_force_buy_vol', 'main_force_sell_vol']
        if intraday_df is not None and not intraday_df.empty and all(c in intraday_df.columns for c in required_cols_4_2):
            mf_buy_vol = intraday_df['main_force_buy_vol'].sum()
            mf_sell_vol = intraday_df['main_force_sell_vol'].sum()
            if (mf_buy_vol + mf_sell_vol) > 0:
                results['main_force_control_leverage'] = ((mf_buy_vol - mf_sell_vol) / (mf_buy_vol + mf_sell_vol)) * 100
        loser_loss_margin = context.get('loser_loss_margin_avg')
        loser_concentration = context.get('loser_concentration_90pct')
        if all(pd.notna(v) for v in [loser_loss_margin, loser_concentration]):
            results['loser_capitulation_pressure_index'] = abs(loser_loss_margin) * loser_concentration
        daily_volume = context.get('daily_turnover_volume')
        if intraday_df is not None and not intraday_df.empty and pd.notna(close_price) and daily_volume and daily_volume > 0:
            new_losers_df = intraday_df[intraday_df['minute_vwap'] > close_price].copy()
            if not new_losers_df.empty:
                new_loser_vol = new_losers_df['vol_shares'].sum()
                if new_loser_vol > 0:
                    new_loser_vwap = (new_losers_df['minute_vwap'] * new_losers_df['vol_shares']).sum() / new_loser_vol
                    avg_loss_rate = abs((close_price / new_loser_vwap - 1))
                    results['intraday_new_loser_pressure'] = (new_loser_vol / daily_volume) * avg_loss_rate * 100
            else:
                results['intraday_new_loser_pressure'] = 0.0
        results['auction_intent_signal'] = 0.0
        results['auction_closing_position'] = 0.0
        if intraday_df is not None and not intraday_df.empty and 'trade_time' in intraday_df.columns and pd.notna(close_price) and pd.notna(atr_14d) and atr_14d > 0:
            auction_start_time = datetime.time(14, 57)
            pre_auction_df = intraday_df[intraday_df['trade_time'].dt.time < auction_start_time]
            auction_df = intraday_df[intraday_df['trade_time'].dt.time >= auction_start_time]
            if not pre_auction_df.empty:
                pre_auction_price = pre_auction_df['close'].iloc[-1]
                if not auction_df.empty and 'vol' in auction_df.columns and auction_df['vol'].sum() > 0:
                    auction_bar = auction_df.iloc[0]
                    auction_volume = auction_bar['vol']
                    if total_daily_vol > 0:
                        price_impact = (close_price - pre_auction_price) / atr_14d
                        volume_weight = np.log1p(auction_volume / total_daily_vol)
                        results['auction_intent_signal'] = price_impact * volume_weight * 100
                    auction_high = auction_bar['high']
                    auction_low = auction_bar['low']
                    auction_range = auction_high - auction_low
                    if auction_range > 1e-6:
                        position = (close_price - auction_low) / auction_range
                        results['auction_closing_position'] = (position * 2 - 1) * 100
                    elif close_price >= auction_high:
                        results['auction_closing_position'] = 100.0
                    else:
                        results['auction_closing_position'] = -100.0
        low_price = context.get('low_price')
        high_price = context.get('high_price')
        if intraday_df is not None and not intraday_df.empty and all(pd.notna(v) for v in [low_price, high_price, close_price]):
            day_range = high_price - low_price
            if day_range > 1e-6:
                if 'low' in intraday_df.columns and not intraday_df['low'].empty:
                    low_price_idx = intraday_df['low'].idxmin()
                else:
                    low_price_idx = intraday_df['minute_vwap'].idxmin()
                rebound_df = intraday_df.loc[low_price_idx:]
                if not rebound_df.empty and len(rebound_df) > 1:
                    price_recovery_ratio = (close_price - low_price) / day_range
                    rebound_volume = rebound_df['vol_shares'].sum()
                    if rebound_volume > 0 and 'main_force_net_vol' in rebound_df.columns:
                        rebound_mf_net_flow = rebound_df['main_force_net_vol'].sum()
                        rebound_purity = rebound_mf_net_flow / rebound_volume
                        results['intraday_probe_rebound_quality'] = price_recovery_ratio * rebound_purity * 100
        return results

    def _compute_vital_sign_metrics(self, context: dict) -> dict:
        """
        【V3.3 · 共识度鲁棒性增强版】
        - 核心修复: 增强 `structural_consensus_score` 的计算鲁棒性，当上游集中度指标为空时，采用中性值替代。
        """
        results = {}
        # 1. 成本结构共识与筹码成本动量
        concentration_90pct = context.get('concentration_90pct')
        total_winner_rate = context.get('total_winner_rate')
        # 增强鲁棒性：如果集中度指标为空，则赋予一个中性值，而不是让计算失败
        if pd.notna(total_winner_rate):
            if pd.notna(concentration_90pct):
                concentration_factor = 1.0 - np.clip(concentration_90pct, 0, 1)
            else:
                # 当上游集中度为空时（例如市场极端行情），赋予一个中性值0.5
                concentration_factor = 0.5
            profit_factor = total_winner_rate / 100.0
            results['structural_consensus_score'] = concentration_factor * profit_factor * 100
        # 升级为 `dominant_cost_momentum`
        dominant_peak_cost = context.get('dominant_peak_cost')
        prev_dominant_peak_cost = context.get('prev_dominant_peak_cost')
        prev_atr = context.get('prev_atr_14d')
        if all(pd.notna(v) for v in [dominant_peak_cost, prev_dominant_peak_cost, prev_atr]) and prev_atr > 0:
            results['dominant_cost_momentum'] = (dominant_peak_cost - prev_dominant_peak_cost) / prev_atr
        # 2. 结构韧性指数
        def normalize(value, default=0.0): return value if pd.notna(value) else default
        consensus_index = normalize(results.get('structural_consensus_score'))
        peak_profit_margin = normalize(context.get('dominant_peak_profit_margin'))
        foundation_score = np.log1p(consensus_index) * np.log1p(np.maximum(0, peak_profit_margin))
        active_winner_margin = normalize(context.get('active_winner_profit_margin'))
        winner_conviction = normalize(context.get('winner_conviction_index'))
        pressure_score = np.log1p(np.maximum(0, active_winner_margin)) * np.log1p(np.maximum(0, winner_conviction))
        cost_momentum = normalize(results.get('dominant_cost_momentum'))
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

    def _compute_static_structure_metrics(self) -> dict:
        """
        【V11.5 · 价格成交量熵集成版】
        - 核心新增: 计算并存储 `price_volume_entropy`。
        - 核心优化: 为 `chip_fault_blockage_ratio` 增加默认值0.0，确保在无断层时该指标有明确输出，而非NULL。
        - 【修复】重新定义 `active_winner_rate` 和 `active_loser_rate` 的计算逻辑，使其在价格突破近期高点时仍能有效捕捉活跃筹码。
        - 【修正】优化 `cost_structure_skewness` 的计算，确保在极端情况下能得到合理值。
        """
        results = {}
        close_price = self.ctx.get('close_price')
        atr_14d = self.ctx.get('atr_14d')
        # 1. 主导峰与潜在次峰引力点(PSGP)剖面
        peaks, properties = find_peaks(self.df['percent'], prominence=0.1, width=1)
        results.update({
            'dominant_peak_cost': np.nan, 'dominant_peak_volume_ratio': np.nan,
            'secondary_peak_cost': np.nan, 'peak_separation_ratio': np.nan,
            'peak_volume_ratio': np.nan, 'peak_distance_volatility_ratio': np.nan,
            'peak_separation_intensity': np.nan, 'peak_fusion_indicator': np.nan
        })
        if not self.df.empty:
            if len(peaks) > 0:
                peaks_df = pd.DataFrame({
                    'peak_index': peaks, 'volume': self.df['percent'].iloc[peaks].values,
                    'cost': self.df['price'].iloc[peaks].values, 'prominence': properties['prominences'],
                    'left_ip': properties['left_ips'], 'right_ip': properties['right_ips'],
                }).sort_values(by='prominence', ascending=False).reset_index(drop=True)
                main_peak = peaks_df.iloc[0]
                main_peak_cost = main_peak['cost']
                main_peak_idx = int(main_peak['peak_index'])
                results['dominant_peak_cost'] = main_peak_cost
                results['dominant_peak_volume_ratio'] = main_peak['volume']
                results['peak_range_low'] = np.interp(main_peak['left_ip'], self.df.index, self.df['price'])
                results['peak_range_high'] = np.interp(main_peak['right_ip'], self.df.index, self.df['price'])
            else:
                main_peak_idx = self.df['percent'].idxmax()
                main_peak_cost = self.df.loc[main_peak_idx, 'price']
                results['dominant_peak_cost'] = main_peak_cost
                results['dominant_peak_volume_ratio'] = self.df.loc[main_peak_idx, 'percent']
                results['peak_range_low'] = main_peak_cost * 0.99
                results['peak_range_high'] = main_peak_cost * 1.01
            peak_width = max(1, int(len(self.df) * 0.05))
            exclusion_start = max(0, main_peak_idx - peak_width)
            exclusion_end = min(len(self.df), main_peak_idx + peak_width)
            remaining_chips_df = self.df.drop(self.df.index[exclusion_start:exclusion_end])
            if not remaining_chips_df.empty:
                psgp_idx = remaining_chips_df['percent'].idxmax()
                psgp_cost = remaining_chips_df.loc[psgp_idx, 'price']
                psgp_volume = remaining_chips_df.loc[psgp_idx, 'percent']
                results['secondary_peak_cost'] = psgp_cost
                if pd.notna(main_peak_cost) and main_peak_cost > 0:
                    separation = abs(main_peak_cost - psgp_cost) / main_peak_cost
                    results['peak_separation_ratio'] = separation * 100
                    valley_df = self.df.loc[min(main_peak_idx, psgp_idx):max(main_peak_idx, psgp_idx)]
                    valley_chip_percent = valley_df['percent'].sum() - results['dominant_peak_volume_ratio'] - psgp_volume
                    fusion_penalty = np.tanh(valley_chip_percent / 10)
                    results['peak_separation_intensity'] = results['peak_separation_ratio'] * (1 - fusion_penalty)
                    results['peak_fusion_indicator'] = (1 - separation) * (1 + fusion_penalty) * 100
                if pd.notna(results['dominant_peak_volume_ratio']) and results['dominant_peak_volume_ratio'] > 0:
                    results['peak_volume_ratio'] = (psgp_volume / results['dominant_peak_volume_ratio']) * 100
                if pd.notna(atr_14d) and atr_14d > 0:
                    peak_distance = abs(main_peak_cost - psgp_cost)
                    results['peak_distance_volatility_ratio'] = peak_distance / atr_14d
        if pd.notna(close_price) and results.get('dominant_peak_cost', 0) > 0:
            results['dominant_peak_profit_margin'] = (close_price / results['dominant_peak_cost'] - 1) * 100
        dominant_peak_cost = results.get('dominant_peak_cost')
        cost_95pct = self.ctx.get('cost_95pct')
        cost_5pct = self.ctx.get('cost_5pct')
        if all(pd.notna(v) for v in [dominant_peak_cost, cost_95pct, cost_5pct, close_price]) and close_price > 0:
            peak_zone_low = dominant_peak_cost * 0.995
            peak_zone_high = dominant_peak_cost * 1.005
            volume_in_peak_zone = self.df[(self.df['price'] >= peak_zone_low) & (self.df['price'] <= peak_zone_high)]['percent'].sum()
            chip_width_90pct = cost_95pct - cost_5pct
            if chip_width_90pct > 0:
                peak_top_density = volume_in_peak_zone / (peak_zone_high - peak_zone_low)
                normalized_chip_width = chip_width_90pct / close_price
                if normalized_chip_width > 0:
                    results['dominant_peak_solidity'] = peak_top_density / normalized_chip_width
        # 2. 集中度剖面
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
        # 3. 动态压力支撑 (升级为力矩模型)
        if pd.notna(close_price) and pd.notna(atr_14d) and atr_14d > 0:
            zone_width = 0.5 * atr_14d
            pressure_df = self.df[(self.df['price'] > close_price) & (self.df['price'] <= close_price + zone_width)]
            if not pressure_df.empty:
                pressure_torque = ((pressure_df['price'] - close_price) / atr_14d * pressure_df['percent']).sum()
                results['dynamic_pressure_index'] = pressure_torque
            support_df = self.df[(self.df['price'] >= close_price - zone_width) & (self.df['price'] < close_price)]
            if not support_df.empty:
                support_torque = ((close_price - support_df['price']) / atr_14d * support_df['percent']).sum()
                results['dynamic_support_index'] = support_torque
        # 4. 盈亏结构
        if pd.notna(close_price):
            winners_df = self.df[self.df['price'] < close_price]
            losers_df = self.df[self.df['price'] > close_price]
            results['total_winner_rate'] = winners_df['percent'].sum()
            results['total_loser_rate'] = losers_df['percent'].sum()
            if not winners_df.empty and winners_df['percent'].sum() > 0:
                winner_avg_cost = np.average(winners_df['price'], weights=winners_df['percent'])
                if winner_avg_cost > 0:
                    results['winner_profit_margin_avg'] = (close_price / winner_avg_cost - 1) * 100
                    if pd.notna(results['total_winner_rate']) and pd.notna(results['winner_profit_margin_avg']):
                        results['effective_winner_rate'] = results['total_winner_rate'] * results['winner_profit_margin_avg']
            if not losers_df.empty and losers_df['percent'].sum() > 0:
                loser_avg_cost = np.average(losers_df['price'], weights=losers_df['percent'])
                if loser_avg_cost > 0: results['loser_loss_margin_avg'] = (close_price / loser_avg_cost - 1) * 100
            if pd.notna(atr_14d) and atr_14d > 0:
                active_winner_mask = (self.df['price'] < close_price) & (self.df['price'] >= close_price - 2 * atr_14d)
                results['active_winner_rate'] = self.df[active_winner_mask]['percent'].sum()
                active_loser_mask = (self.df['price'] > close_price) & (self.df['price'] <= close_price + 2 * atr_14d)
                results['active_loser_rate'] = self.df[active_loser_mask]['percent'].sum()
            else:
                results['active_winner_rate'] = 0.0
                results['active_loser_rate'] = 0.0
            locked_profit_mask = (self.df['price'] < close_price) & (~active_winner_mask if pd.notna(atr_14d) else (self.df['price'] < close_price))
            results['locked_profit_rate'] = self.df[locked_profit_mask]['percent'].sum()
            locked_loss_mask = (self.df['price'] > close_price) & (~active_loser_mask if pd.notna(atr_14d) else (self.df['price'] > close_price))
            results['locked_loss_rate'] = self.df[locked_loss_mask]['percent'].sum()
        # 5. 断层动态 (鲁棒性优化)
        peak_cost = results.get('dominant_peak_cost')
        results['chip_fault_blockage_ratio'] = 0.0
        if pd.notna(peak_cost) and pd.notna(close_price) and pd.notna(atr_14d) and atr_14d > 0:
            magnitude = (close_price - peak_cost) / atr_14d
            results['chip_fault_magnitude'] = magnitude
            fault_zone_low, fault_zone_high = sorted([peak_cost, close_price])
            fault_zone_df = self.df[(self.df['price'] > fault_zone_low) & (self.df['price'] < fault_zone_high)]
            if not fault_zone_df.empty:
                results['chip_fault_blockage_ratio'] = fault_zone_df['percent'].sum()
        # 6. 分层成本
        high_20d, low_20d = self.ctx.get('high_20d'), self.ctx.get('low_20d')
        if pd.notna(high_20d) and pd.notna(low_20d):
            active_zone_mask = (self.df['price'] >= low_20d) & (self.df['price'] <= high_20d)
            short_term_chips_df = self.df[active_zone_mask]
            long_term_chips_df = self.df[~active_zone_mask]
            if not short_term_chips_df.empty and short_term_chips_df['percent'].sum() > 0:
                results['short_term_holder_cost'] = np.average(short_term_chips_df['price'], weights=short_term_chips_df['percent'])
                results['short_term_concentration_90pct'] = _get_concentration(short_term_chips_df)
            if not long_term_chips_df.empty and long_term_chips_df['percent'].sum() > 0:
                results['long_term_holder_cost'] = np.average(long_term_chips_df['price'], weights=long_term_chips_df['percent'])
                results['long_term_concentration_90pct'] = _get_concentration(long_term_chips_df)
        # 7. 盈利亏损质量分析
        if not self.df.empty and pd.notna(close_price):
            winners_df_quality = self.df[self.df['price'] < close_price].copy()
            if not winners_df_quality.empty and winners_df_quality['percent'].sum() > 0:
                winners_df_quality['percent'] = (winners_df_quality['percent'] / winners_df_quality['percent'].sum()) * 100
                winners_df_quality['cum_percent'] = winners_df_quality['percent'].cumsum()
                cost_at_85pct = np.interp(85, winners_df_quality['cum_percent'], winners_df_quality['price'])
                if pd.notna(cost_at_85pct) and cost_at_85pct > 0:
                    results['winner_profit_cushion'] = (close_price / cost_at_85pct - 1) * 100
            losers_df_quality = self.df[self.df['price'] > close_price].copy()
            if not losers_df_quality.empty and losers_df_quality['percent'].sum() > 0:
                loser_avg_cost_quality = np.average(losers_df_quality['price'], weights=losers_df_quality['percent'])
                avg_loss_margin = abs((close_price / loser_avg_cost_quality - 1) * 100) if loser_avg_cost_quality > 0 else 0
                losers_df_quality['percent'] = (losers_df_quality['percent'] / losers_df_quality['percent'].sum()) * 100
                losers_df_quality['cum_percent'] = losers_df_quality['percent'].cumsum()
                cost_5pct_loser = np.interp(5, losers_df_quality['cum_percent'], losers_df_quality['price'])
                cost_95pct_loser = np.interp(95, losers_df_quality['cum_percent'], losers_df_quality['price'])
                loser_concentration = (cost_95pct_loser - cost_5pct_loser) / loser_avg_cost_quality if loser_avg_cost_quality > 0 else 0
                results['loser_pain_index'] = avg_loss_margin * (1 - np.clip(loser_concentration, 0, 1))
        # 8. 成本分布形态分析
        results['cost_structure_skewness'] = self._calculate_cost_structure_skewness(self.ctx)
        # 9. 结构稳定性评估
        concentration_70pct = self.ctx.get('concentration_70pct')
        total_winner_rate_stability = results.get('total_winner_rate')
        winner_profit_cushion_stability = results.get('winner_profit_cushion')
        dominant_peak_profit_margin_stability = results.get('dominant_peak_profit_margin')
        if all(pd.notna(v) for v in [concentration_70pct, total_winner_rate_stability, winner_profit_cushion_stability, dominant_peak_profit_margin_stability]):
            concentration_score = np.exp(-5 * np.clip(concentration_70pct, 0, None))
            winner_rate_score = total_winner_rate_stability / 100.0
            cushion_score = np.tanh(winner_profit_cushion_stability / 10.0)
            peak_profit_score = np.tanh(dominant_peak_profit_margin_stability / 20.0)
            scores = [s for s in [concentration_score, winner_rate_score, cushion_score, peak_profit_score] if s > 0]
            if scores:
                stability_raw = np.prod(scores)
                final_score = stability_raw ** (1.0 / len(scores))
                results['structural_stability_score'] = final_score * 100
        # 10. 近期套牢盘压力
        recent_5d_high = self.ctx.get('high_5d')
        recent_5d_low = self.ctx.get('low_5d')
        turnover_vol_5d = self.ctx.get('turnover_vol_5d')
        if all(pd.notna(v) for v in [recent_5d_high, recent_5d_low, turnover_vol_5d, close_price]) and turnover_vol_5d > 0:
            trapped_mask = (self.df['price'] > close_price) & (self.df['price'] >= recent_5d_low) & (self.df['price'] <= recent_5d_high)
            recent_trapped_percent = self.df[trapped_mask]['percent'].sum()
            total_chip_volume = self.ctx.get('total_chip_volume', 0)
            if total_chip_volume > 0:
                recent_trapped_vol = (recent_trapped_percent / 100) * total_chip_volume
                results['recent_trapped_pressure'] = (recent_trapped_vol / turnover_vol_5d) * 100
        # 11. 潜在获利盘供给
        if pd.notna(close_price):
            imminent_supply_mask = (self.df['price'] >= close_price / 1.05) & (self.df['price'] < close_price)
            results['imminent_profit_taking_supply'] = self.df[imminent_supply_mask]['percent'].sum()
        # 12. 价格成交量熵 (新增)
        intraday_df = self.ctx.get('processed_intraday_df')
        daily_high = self.ctx.get('high_price')
        daily_low = self.ctx.get('low_price')
        total_daily_volume = self.ctx.get('daily_turnover_volume')
        results['price_volume_entropy'] = self._calculate_price_volume_entropy(intraday_df, daily_high, daily_low, total_daily_volume)
        return results

    def _compute_intraday_dynamics_metrics(self, ctx: dict) -> dict:
        """
        【V1.12 · 属性访问修正】计算日内动态指标。
        - 核心修复: 修正调试信息中对 `stock_code` 属性的访问方式，从 `self.stock_code` 改为 `ctx.get('stock_code')`。
        - 核心修复: 使用 `ctx.get()` 安全访问 `minute_data_for_day`，避免 `KeyError`。
        - 核心修复: 确保 `intraday_df` 始终为 DataFrame 类型，即使数据缺失也能正常处理。
        """
        # 调试信息: 打印 ctx 字典的键，以帮助诊断问题
        # 修正 stock_code 的访问方式
        # 使用 .get() 方法安全访问键，提供默认值 pd.DataFrame()
        intraday_df = ctx.get('minute_data_for_day', pd.DataFrame())
        if intraday_df.empty:
            print(f"调试信息: [{ctx.get('stock_code', 'UNKNOWN')}] _compute_intraday_dynamics_metrics - minute_data_for_day 为空，跳过计算。")
            return {}
        metrics = {}
        # 确保索引是 DatetimeIndex
        if not isinstance(intraday_df.index, pd.DatetimeIndex):
            logger.error(f"[{ctx.get('stock_code', 'UNKNOWN')}] 日内数据索引不是 DatetimeIndex，无法计算日内动态指标。")
            return {}
        # 使用 intraday_df.index 访问时间
        opening_30min_df = intraday_df[intraday_df.index.time < pd.to_datetime('10:00').time()]
        if not opening_30min_df.empty:
            metrics['opening_30min_vol_ratio'] = opening_30min_df['vol_shares'].sum() / intraday_df['vol_shares'].sum()
            metrics['opening_30min_range_ratio'] = (opening_30min_df['high'].max() - opening_30min_df['low'].min()) / (intraday_df['high'].max() - intraday_df['low'].min())
            metrics['opening_30min_vwap_change'] = (opening_30min_df['minute_vwap'].iloc[-1] - opening_30min_df['minute_vwap'].iloc[0]) / opening_30min_df['minute_vwap'].iloc[0]
        # 使用 intraday_df.index 访问时间
        closing_30min_df = intraday_df[intraday_df.index.time >= pd.to_datetime('14:30').time()]
        if not closing_30min_df.empty:
            metrics['closing_30min_vol_ratio'] = closing_30min_df['vol_shares'].sum() / intraday_df['vol_shares'].sum()
            metrics['closing_30min_range_ratio'] = (closing_30min_df['high'].max() - closing_30min_df['low'].min()) / (intraday_df['high'].max() - intraday_df['low'].min())
            metrics['closing_30min_vwap_change'] = (closing_30min_df['minute_vwap'].iloc[-1] - closing_30min_df['minute_vwap'].iloc[0]) / closing_30min_df['minute_vwap'].iloc[0]
        # 计算日内波动率
        metrics['intraday_volatility'] = intraday_df['minute_vwap'].std() / intraday_df['minute_vwap'].mean()
        # 计算日内成交量分布的偏度
        metrics['intraday_volume_skew'] = intraday_df['vol_shares'].skew()
        # 计算日内价格趋势的持续性
        price_diff = intraday_df['minute_vwap'].diff().dropna()
        metrics['intraday_trend_persistence'] = (price_diff > 0).sum() / len(price_diff) if len(price_diff) > 0 else 0.5
        # 计算日内价格反转强度
        metrics['intraday_reversal_strength'] = (intraday_df['close'].iloc[-1] - intraday_df['open'].iloc[0]) / (intraday_df['high'].max() - intraday_df['low'].min())
        # 计算日内成交量加权平均价（VWAP）与收盘价的偏离
        metrics['vwap_close_deviation'] = (intraday_df['close'].iloc[-1] - intraday_df['minute_vwap'].mean()) / intraday_df['minute_vwap'].mean()
        # 计算日内高低点与收盘价的相对位置
        metrics['close_to_range_ratio'] = (intraday_df['close'].iloc[-1] - intraday_df['low'].min()) / (intraday_df['high'].max() - intraday_df['low'].min())
        return metrics

    def _calculate_chip_structure_health_score(self, context: dict) -> dict:
        """
        【V4.2 · 生产就绪版】
        - 核心优化: 移除所有调试探针，代码恢复生产状态。
        """
        from scipy.stats import percentileofscore
        results = {'chip_health_score': np.nan} # 默认值为NaN
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
                    continue # 如果当前值或历史数据缺失，则安全跳过此子指标
                historical_series = historical_df[metric].dropna()
                if historical_series.empty:
                    continue
                percentile = percentileofscore(historical_series, current_value, kind='rank') / 100.0
                normalized_score = percentile if weight > 0 else (1.0 - percentile)
                component_scores.append(normalized_score)
            # 增强鲁棒性：如果维度有有效分数，则计算其平均分；否则赋予中性分
            if component_scores:
                dimension_scores[dim_name] = np.mean(component_scores)
            else:
                # 如果该维度的所有子指标都无法计算，则给予一个中性的0.5分（50分位），避免整个健康分失败
                dimension_scores[dim_name] = 0.5
                logger.debug(f"[{context.get('stock_code')}] [{context.get('trade_date')}] 健康分维度 '{dim_name}' 无法计算，赋予中性分0.5。")
        # 4. 使用几何平均整合各维度得分
        if not dimension_scores:
            return results
        final_score_raw = 1.0
        valid_dims = 0
        for score in dimension_scores.values():
            if pd.notna(score):
                final_score_raw *= score
                valid_dims += 1
        if valid_dims > 0:
            final_score_normalized = final_score_raw ** (1.0 / valid_dims)
            results['chip_health_score'] = final_score_normalized * 100
        return results

    def _calculate_active_winner_profit_margin(self, close_price: float, atr_14d: float, context: dict) -> tuple[float, float]:
        """
        【V1.0】计算活跃获利盘利润率，并加入探针。
        """
        stock_code = context.get('stock_code', 'UNKNOWN')
        trade_date = context.get('trade_date', 'UNKNOWN')
        debug_params = context.get('debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        is_probe_date = False
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0]).date()
            if probe_date_naive == trade_date:
                is_probe_date = True
        active_winner_avg_cost = np.nan
        active_profit_margin = 0.0
        if pd.notna(close_price) and pd.notna(atr_14d) and atr_14d > 0:
            # 活跃获利盘：成本在 (close_price - 3*ATR, close_price) 之间，扩大范围
            active_winners_df = self.df[(self.df['price'] < close_price) & (self.df['price'] >= close_price - 3 * atr_14d)]
            if is_probe_date:
                print(f"    -> [活跃获利盘利润率探针] @ {trade_date}:")
                print(f"       - close_price: {close_price:.4f}, atr_14d: {atr_14d:.4f}")
                print(f"       - active_winners_df (filtered): {active_winners_df.head()}")
                print(f"       - active_winners_df.empty: {active_winners_df.empty}, percent.sum(): {active_winners_df['percent'].sum():.4f}")
            if not active_winners_df.empty and active_winners_df['percent'].sum() > 0:
                active_winner_avg_cost = np.average(active_winners_df['price'], weights=active_winners_df['percent'])
                if active_winner_avg_cost > 0:
                    active_profit_margin = ((close_price - active_winner_avg_cost) / active_winner_avg_cost) * 100
                if is_probe_date:
                    print(f"       - 活跃获利盘存在。active_winner_avg_cost: {active_winner_avg_cost:.4f}, active_profit_margin: {active_profit_margin:.4f}")
            else:
                # 如果活跃获利盘区间为空，则考虑更广阔的获利盘范围，例如使用 dominant_peak_cost
                dominant_peak_cost = context.get('dominant_peak_cost')
                if is_probe_date:
                    print(f"       - 活跃获利盘区间为空。尝试使用 dominant_peak_cost: {dominant_peak_cost:.4f}")
                if pd.notna(dominant_peak_cost) and dominant_peak_cost > 0 and dominant_peak_cost < close_price:
                    active_winner_avg_cost = dominant_peak_cost
                    active_profit_margin = ((close_price - dominant_peak_cost) / dominant_peak_cost) * 100
                    if is_probe_date:
                        print(f"       - 使用 dominant_peak_cost。active_winner_avg_cost: {active_winner_avg_cost:.4f}, active_profit_margin: {active_profit_margin:.4f}")
                else:
                    # 在涨停日，如果无法计算出活跃获利盘，则赋予一个积极的默认利润率
                    # 假设涨停日至少有 9% 的利润率
                    pre_close = context.get('pre_close', close_price)
                    if (close_price / pre_close - 1) > 0.098: # 如果是涨停日 (考虑0.098作为涨停阈值)
                        active_profit_margin = 9.0 # 涨停日默认利润率
                        if is_probe_date:
                            print(f"       - 涨停日，赋予默认利润率: {active_profit_margin:.4f}")
                    else:
                        active_profit_margin = 0.0 # 默认利润率为0
                        if is_probe_date:
                            print(f"       - 非涨停日且无活跃获利盘，利润率: {active_profit_margin:.4f}")
        else:
            if is_probe_date:
                print(f"       - close_price 或 atr_14d 无效，利润率: {active_profit_margin:.4f}")
        return active_winner_avg_cost, active_profit_margin

    def _calculate_winner_conviction_index(self, context: dict, active_profit_margin: float) -> float:
        """
        【V1.5 · 压力缓和器版】计算赢家信念指数。
        - 核心升级: 将 `realized_pressure` 的计算函数从 `tanh` 替换为 `arctan`。
        - 核心思想: 使用 `arctan` 作为“压力缓和器”，解决因分母过小（小基数陷阱）导致的 `pressure_ratio` 异常放大问题，增强指标对极端值的鲁棒性。
        """
        stock_code = context.get('stock_code', 'UNKNOWN')
        trade_date = context.get('trade_date', 'UNKNOWN')
        debug_params = context.get('debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        is_probe_date = False
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0]).date()
            if probe_date_naive == trade_date:
                is_probe_date = True
        bullish_reinforcement = context.get('upward_impulse_purity', 0.0)
        profit_taking_flow_ratio = context.get('profit_taking_flow_ratio', 0.0)
        active_winner_rate = context.get('active_winner_rate', 0.0)
        close_price = context.get('close_price')
        pre_close = context.get('pre_close', close_price)
        winner_conviction_index = 0.0
        if is_probe_date:
            print(f"    -> [赢家信念指数探针] @ {trade_date}:")
            print(f"       - active_profit_margin: {active_profit_margin:.4f}")
            print(f"       - bullish_reinforcement: {bullish_reinforcement:.4f}")
            print(f"       - profit_taking_flow_ratio: {profit_taking_flow_ratio:.4f}")
            print(f"       - active_winner_rate: {active_winner_rate:.4f}")
        if all(pd.notna(v) for v in [active_profit_margin, bullish_reinforcement, profit_taking_flow_ratio, active_winner_rate]):
            pressure_ratio = 0.0
            realized_pressure = 0.0
            if active_winner_rate > 1e-6: # 增加一个极小值判断，避免除零
                pressure_ratio = (profit_taking_flow_ratio / 100.0) / (active_winner_rate / 100.0)
                # 修改代码行: 使用 arctan 替换 tanh，增强鲁棒性
                realized_pressure = (2 / np.pi) * np.arctan(np.clip(pressure_ratio - 1.0, 0, None))
            hesitation_factor = 1.0 + (1.0 - realized_pressure)
            margin_factor = np.log1p(np.clip(active_profit_margin / 100.0, 0, None)) if active_profit_margin > 0 else 0.0
            reinforcement_factor = np.exp(bullish_reinforcement / 100.0)
            winner_conviction_index = hesitation_factor * margin_factor * reinforcement_factor * 100
            if (close_price / pre_close - 1) > 0.098 and active_profit_margin > 0:
                winner_conviction_index = np.maximum(winner_conviction_index, 10.0)
            if is_probe_date:
                print(f"       - pressure_ratio: {pressure_ratio:.4f}")
                print(f"       - realized_pressure (arctan): {realized_pressure:.4f}")
                print(f"       - hesitation_factor: {hesitation_factor:.4f}")
                print(f"       - margin_factor: {margin_factor:.4f}")
                print(f"       - reinforcement_factor: {reinforcement_factor:.4f}")
                print(f"       - final winner_conviction_index: {winner_conviction_index:.4f}")
        else:
            if is_probe_date:
                print(f"       - 缺少前置条件，winner_conviction_index: {winner_conviction_index:.4f}")
        return winner_conviction_index

    def _calculate_cost_structure_skewness(self, context: dict) -> float:
        """
        【V1.1】计算成本结构偏度，并加入探针。
        - 【修正】移除对 `skewness` 的负号操作，使正偏度（筹码集中在高价区）对应正值，符合涨停日积极信号的预期。
        """
        stock_code = context.get('stock_code', 'UNKNOWN')
        trade_date = context.get('trade_date', 'UNKNOWN')
        debug_params = context.get('debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        is_probe_date = False
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0]).date()
            if probe_date_naive == trade_date:
                is_probe_date = True
        skewness = 0.0
        if not self.df.empty and self.df['percent'].sum() >= 1e-6:
            from scipy.stats import skew
            total_percent = self.df['percent'].sum()
            weights = np.round((self.df['percent'] / total_percent) * 10000).astype(int)
            valid_weights = weights[weights > 0]
            if is_probe_date:
                print(f"    -> [成本结构偏度探针] @ {trade_date}:")
                print(f"       - total_percent: {total_percent:.4f}")
                print(f"       - valid_weights count: {len(valid_weights)}")
            if len(valid_weights) >= 3: # 至少需要3个点才能计算偏度
                valid_prices = self.df['price'][weights > 0]
                unweighted_sample = np.repeat(valid_prices, valid_weights)
                skewness = skew(unweighted_sample)
                # 移除负号操作，使正偏度（筹码集中在高价区）对应正值
                # skewness = -skewness
                if is_probe_date:
                    print(f"       - unweighted_sample min/max: {np.min(unweighted_sample):.4f}/{np.max(unweighted_sample):.4f}")
                    print(f"       - raw skewness: {skewness:.4f}, final cost_structure_skewness: {skewness:.4f}")
            else:
                if is_probe_date:
                    print(f"       - 有效权重不足3个，无法计算偏度。skewness: {skewness:.4f}")
        else:
            if is_probe_date:
                print(f"       - 筹码分布为空或百分比和过低，skewness: {skewness:.4f}")
        return skewness

    def _calculate_price_volume_entropy(self, intraday_df: pd.DataFrame, daily_high: float, daily_low: float, total_daily_volume: float) -> float:
        """
        【V1.0】计算价格成交量熵。
        - 核心思想: 将日内价格区间划分为若干价格箱，计算每个价格箱内的成交量比例，然后计算其香农熵。
        - 意义: 量化日内交易活动在不同价格区间的集中度。低熵值表示交易集中，高熵值表示交易分散。
        """
        from scipy.stats import entropy
        stock_code = self.ctx.get('stock_code', 'UNKNOWN')
        trade_date = self.ctx.get('trade_date', 'UNKNOWN')
        if intraday_df.empty or total_daily_volume <= 0 or pd.isna(daily_high) or pd.isna(daily_low) or daily_high <= daily_low:
            # print(f"调试信息: [{stock_code}] [{trade_date}] 价格成交量熵计算跳过，原因：日内数据为空或总成交量为零或价格范围无效。")
            return np.nan
        # 确定价格箱的数量
        # 动态确定价格箱数量，基于最小价格变动单位0.01元，并设置上下限
        price_range = daily_high - daily_low
        if price_range <= 0.01: # 如果价格几乎没有波动，至少给2个箱子
            num_bins = 2
        else:
            num_bins = int(price_range / 0.01) + 1 # 每个0.01元一个箱子
            num_bins = np.clip(num_bins, 20, 200) # 设置最小20个，最大200个箱子
        # 使用 minute_vwap 作为价格，vol_shares 作为成交量进行分箱
        # 确保价格数据是数值类型
        prices = pd.to_numeric(intraday_df['minute_vwap'], errors='coerce')
        volumes = pd.to_numeric(intraday_df['vol_shares'], errors='coerce')
        # 过滤掉无效数据
        valid_data = pd.DataFrame({'price': prices, 'volume': volumes}).dropna()
        if valid_data.empty or valid_data['volume'].sum() <= 0:
            print(f"调试信息: [{stock_code}] [{trade_date}] 价格成交量熵计算跳过，原因：有效日内数据为空或总成交量为零。")
            return np.nan
        # 使用 pd.cut 进行分箱，duplicates='drop' 处理价格无波动的情况
        bins = pd.cut(valid_data['price'], bins=num_bins, include_lowest=True, duplicates='drop')
        # 统计每个价格箱内的成交量
        volume_per_bin = valid_data.groupby(bins)['volume'].sum()
        # 过滤掉没有成交量的箱子
        volume_per_bin = volume_per_bin[volume_per_bin > 0]
        if volume_per_bin.empty:
            print(f"调试信息: [{stock_code}] [{trade_date}] 价格成交量熵计算结果：0 (所有成交量集中在单个价格点或无成交)。")
            return 0.0 # 所有成交量集中在单个价格点，熵为0
        # 计算每个价格箱的成交量比例 (p_i)
        probabilities = volume_per_bin / volume_per_bin.sum()
        # 计算香农熵 (以2为底)
        shannon_entropy = entropy(probabilities, base=2)
        # 归一化熵值 (可选，但推荐，使其在0到1之间)
        max_entropy = np.log2(len(volume_per_bin)) if len(volume_per_bin) > 1 else 0
        normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0.0
        print(f"调试信息: [{stock_code}] [{trade_date}] 价格成交量熵计算结果：{normalized_entropy:.4f} (原始熵: {shannon_entropy:.4f}, 箱数: {len(volume_per_bin)})。")
        return normalized_entropy

    def _compute_microstructure_game_metrics(self, context: dict) -> dict:
        """
        【V1.1 · 健壮回退版】计算基于高频数据的筹码微观博弈指标。
        - 核心增强: 增加数据源检查。如果日内数据不包含高频特征（如 'buy_vol_raw'），则优雅地跳过计算，返回默认值。
        """
        results = {
            'peak_exchange_purity': np.nan,
            'pressure_validation_score': np.nan,
            'support_validation_score': np.nan,
            'covert_accumulation_signal': np.nan,
        }
        intraday_df = context.get('processed_intraday_df')
        # 核心增强：检查是否存在高频数据列，如果不存在则直接返回NaN
        if intraday_df.empty or 'buy_vol_raw' not in intraday_df.columns:
            return results
        peak_low = context.get('peak_range_low')
        peak_high = context.get('peak_range_high')
        if not all(pd.notna(v) for v in [peak_low, peak_high]):
            return results
        # 1. 筹码交换纯度 (Peak Exchange Purity)
        peak_zone_df = intraday_df[(intraday_df['minute_vwap'] >= peak_low) & (intraday_df['minute_vwap'] <= peak_high)]
        if not peak_zone_df.empty:
            total_vol_peak = peak_zone_df['vol_shares'].sum()
            if total_vol_peak > 0:
                active_buy_vol = peak_zone_df['buy_vol_raw'].sum()
                active_sell_vol = peak_zone_df['sell_vol_raw'].sum()
                purity = 1 - abs(active_buy_vol - active_sell_vol) / total_vol_peak
                results['peak_exchange_purity'] = purity * 100
        # 2. 压力/支撑有效性验证 (Pressure/Support Validation)
        pre_close = context.get('pre_close')
        if pd.notna(pre_close):
            pressure_zone_df = intraday_df[intraday_df['minute_vwap'] > pre_close]
            support_zone_df = intraday_df[intraday_df['minute_vwap'] < pre_close]
            if not pressure_zone_df.empty:
                total_vol_pressure = pressure_zone_df['vol_shares'].sum()
                if total_vol_pressure > 0:
                    active_sell_pressure = pressure_zone_df['sell_vol_raw'].sum()
                    results['pressure_validation_score'] = (active_sell_pressure / total_vol_pressure) * 100
            if not support_zone_df.empty:
                total_vol_support = support_zone_df['vol_shares'].sum()
                if total_vol_support > 0:
                    active_buy_support = support_zone_df['buy_vol_raw'].sum()
                    results['support_validation_score'] = (active_buy_support / total_vol_support) * 100
        # 3. 隐蔽吸筹信号 (Covert Accumulation Signal)
        dip_or_flat_df = intraday_df[intraday_df['close'] <= intraday_df['open']]
        if not dip_or_flat_df.empty:
            total_vol_dip = dip_or_flat_df['vol_shares'].sum()
            if total_vol_dip > 0:
                mf_net_buy_on_dip = dip_or_flat_df['main_force_net_vol'].clip(lower=0).sum()
                results['covert_accumulation_signal'] = (mf_net_buy_on_dip / total_vol_dip) * 100
        return results







