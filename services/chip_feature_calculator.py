# 文件: services/chip_feature_calculator.py

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from decimal import Decimal

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
        """【V20.0 · 筹码迁徙版】"""
        if self.df.empty or not all(k in self.ctx for k in ['weight_avg', 'winner_rate', 'cost_95pct', 'cost_5pct', 'close_price', 'total_chip_volume']):
            return {}
        summary_info = self._get_summary_metrics_from_context()
        self.ctx.update(summary_info)
        peaks_info = self._calculate_peaks()
        concentration_info = self._calculate_concentration_from_perf()
        winner_structure_info = self._calculate_winner_structure()
        holder_costs_info = self._calculate_holder_costs()
        pressure_support_info = self._calculate_pressure_support()
        context_for_derived_metrics = {
            **self.ctx,
            **peaks_info,
            **concentration_info,
            **winner_structure_info,
            **holder_costs_info,
        }
        turnover_microstructure_info = self._calculate_turnover_microstructure(context_for_derived_metrics)
        concentration_dynamics_info = self._calculate_concentration_dynamics(context_for_derived_metrics)
        peak_dynamics_info = self._calculate_peak_dynamics(context_for_derived_metrics)
        minute_derived_dynamics_info = self._calculate_minute_derived_dynamics(context_for_derived_metrics)
        chip_interaction_info = self._calculate_chip_interaction_dynamics(context_for_derived_metrics)
        # 调用新增的跨日筹码流计算方法
        cross_day_flow_info = self._calculate_cross_day_chip_flow(context_for_derived_metrics)
        
        advanced_structure_info = self._calculate_advanced_structures(context_for_derived_metrics)
        fault_info = self._calculate_chip_fault(context_for_derived_metrics)
        all_metrics = {
            **peaks_info,
            **concentration_info,
            **winner_structure_info,
            **holder_costs_info,
            **pressure_support_info,
            **turnover_microstructure_info,
            **concentration_dynamics_info,
            **peak_dynamics_info,
            **minute_derived_dynamics_info,
            **chip_interaction_info,
            # 合并新的跨日筹码流指标
            **cross_day_flow_info,
            
            **advanced_structure_info,
            **fault_info
        }
        if 'total_winner_rate' in summary_info:
            all_metrics['total_winner_rate'] = summary_info['total_winner_rate']
        if 'total_loser_rate' in winner_structure_info:
            all_metrics['total_loser_rate'] = winner_structure_info['total_loser_rate']
        all_metrics.pop('peak_range_low', None)
        all_metrics.pop('peak_range_high', None)
        return all_metrics

    def _prepare_minute_data_features(self):
        """【V2.0 · 内存优化版】对单日分钟数据进行类型降级，减少内存占用。"""
        minute_df = self.ctx.get('minute_data')
        if minute_df is None or minute_df.empty:
            self.ctx.update({'daily_vwap': None, 'volume_above_vwap_ratio': None, 'volume_below_vwap_ratio': None})
            return
        # 内存优化：对列进行类型降级
        # 使用字典进行批量转换和错误处理
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
        
        total_amount_yuan = (minute_df['amount'] * 1000).sum()
        total_vol_shares = (minute_df['vol'] * 100).sum()
        daily_vwap = total_amount_yuan / total_vol_shares if total_vol_shares > 0 else None
        self.ctx['daily_vwap'] = daily_vwap
        if daily_vwap is None:
            self.ctx.update({'volume_above_vwap_ratio': None, 'volume_below_vwap_ratio': None})
            return
        minute_df['minute_vwap'] = ((minute_df['amount'] * 1000) / (minute_df['vol'] * 100).replace(0, np.nan)).astype('float32', errors='ignore')
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

    def _calculate_peaks(self) -> dict:
        """【V10.1 逻辑统一版】"""
        peaks, properties = find_peaks(self.df['percent'], prominence=0.1, width=1)
        def get_price_from_interpolated_index(interp_idx):
            price_indices = self.df.index.to_numpy()
            price_values = self.df['price'].to_numpy()
            return np.interp(interp_idx, price_indices, price_values)
        if len(peaks) == 0:
            peak_idx = self.df['percent'].idxmax()
            peak_price = self.df.loc[peak_idx, 'price']
            peak_range_low = peak_price * 0.995
            peak_range_high = peak_price * 1.005
            return {
                'peak_cost': peak_price,
                'peak_percent': self.df.loc[peak_idx, 'percent'],
                'peak_volume': int(self.df.loc[peak_idx, 'percent'] / 100 * self.ctx['total_chip_volume']),
                'peak_stability': 1.0,
                'is_multi_peak': False,
                'secondary_peak_cost': None,
                'peak_distance_ratio': None,
                'peak_strength_ratio': None,
                'peak_range_low': peak_range_low,
                'peak_range_high': peak_range_high,
            }
        prominences = properties['prominences']
        main_peak_idx_in_peaks = np.argmax(prominences)
        main_peak_df_idx = peaks[main_peak_idx_in_peaks]
        main_peak_cost = self.df.iloc[main_peak_df_idx]['price']
        main_peak_percent = self.df.iloc[main_peak_df_idx]['percent']
        main_peak_volume = int(main_peak_percent / 100 * self.ctx['total_chip_volume'])
        peak_stability = prominences[main_peak_idx_in_peaks] / self.df['percent'].mean() if self.df['percent'].mean() > 0 else 1.0
        left_boundary_idx = properties['left_ips'][main_peak_idx_in_peaks]
        right_boundary_idx = properties['right_ips'][main_peak_idx_in_peaks]
        peak_range_low = get_price_from_interpolated_index(left_boundary_idx)
        peak_range_high = get_price_from_interpolated_index(right_boundary_idx)
        result = {
            'peak_cost': main_peak_cost,
            'peak_percent': main_peak_percent,
            'peak_volume': main_peak_volume,
            'peak_stability': peak_stability,
            'is_multi_peak': len(peaks) > 1,
            'peak_range_low': peak_range_low,
            'peak_range_high': peak_range_high,
        }
        if result['is_multi_peak']:
            remaining_peaks_indices = np.delete(peaks, main_peak_idx_in_peaks)
            remaining_prominences = np.delete(prominences, main_peak_idx_in_peaks)
            secondary_peak_df_idx = remaining_peaks_indices[np.argmax(remaining_prominences)]
            result['secondary_peak_cost'] = self.df.iloc[secondary_peak_df_idx]['price']
            result['peak_distance_ratio'] = abs(main_peak_cost - result['secondary_peak_cost']) / main_peak_cost if main_peak_cost > 0 else None
            result['peak_strength_ratio'] = remaining_prominences.max() / prominences[main_peak_idx_in_peaks] if prominences[main_peak_idx_in_peaks] > 0 else None
        else:
            result.update({'secondary_peak_cost': None, 'peak_distance_ratio': None, 'peak_strength_ratio': None})
        return result

    def _calculate_concentration_from_perf(self) -> dict:
        """【V13.0 重构】直接基于 cyq_perf 提供的成本分位数计算筹码集中度。"""
        cost_95pct = self.ctx.get('cost_95pct')
        cost_5pct = self.ctx.get('cost_5pct')
        cost_85pct = self.ctx.get('cost_85pct')
        cost_15pct = self.ctx.get('cost_15pct')
        weight_avg_cost = self.ctx.get('weight_avg_cost')
        if not all([cost_95pct, cost_5pct, cost_85pct, cost_15pct, weight_avg_cost]) or weight_avg_cost <= 0:
            return {'concentration_90pct': None, 'concentration_70pct': None}
        width_90 = cost_95pct - cost_5pct
        width_70 = cost_85pct - cost_15pct
        self.ctx['cost_range_70pct'] = (cost_15pct, cost_85pct)
        return {
            'concentration_90pct': width_90 / weight_avg_cost,
            'concentration_70pct': width_70 / weight_avg_cost,
        }

    def _calculate_winner_structure(self) -> dict:
        """【V13.6 · 普罗米修斯之火 · 最终正确版】"""
        close_price = self.ctx.get('close_price')
        weight_avg_cost = self.ctx.get('weight_avg_cost')
        total_winner_rate = self.ctx.get('total_winner_rate', 0.0)
        if not all(pd.notna(v) for v in [close_price, weight_avg_cost]):
            return {
                'winner_rate_short_term': None, 'winner_rate_long_term': None,
                'loser_rate_short_term': None, 'loser_rate_long_term': None,
                'total_loser_rate': 100.0 - total_winner_rate if total_winner_rate is not None else None
            }
        winners_df = self.df[self.df['price'] < close_price]
        losers_df = self.df[self.df['price'] > close_price]
        long_term_winner_rate = winners_df[winners_df['price'] < weight_avg_cost]['percent'].sum()
        short_term_winner_rate = winners_df[winners_df['price'] >= weight_avg_cost]['percent'].sum()
        long_term_loser_rate = losers_df[losers_df['price'] < weight_avg_cost]['percent'].sum()
        short_term_loser_rate = losers_df[losers_df['price'] >= weight_avg_cost]['percent'].sum()
        total_loser_rate = long_term_loser_rate + short_term_loser_rate
        return {
            'winner_rate_short_term': short_term_winner_rate,
            'winner_rate_long_term': long_term_winner_rate,
            'loser_rate_short_term': short_term_loser_rate,
            'loser_rate_long_term': long_term_loser_rate,
            'total_loser_rate': total_loser_rate,
        }

    def _calculate_holder_costs(self) -> dict:
        """【V1.1 · 阿瑞斯之盾协议】计算长/短期持仓者平均成本"""
        prev_20d_close = self.ctx.get('prev_20d_close')
        if pd.isna(prev_20d_close):
            return {'avg_cost_short_term': None, 'avg_cost_long_term': None}
        long_term_chips_df = self.df[self.df['price'] < prev_20d_close]
        short_term_chips_df = self.df[self.df['price'] >= prev_20d_close]
        if not long_term_chips_df.empty and long_term_chips_df['percent'].sum() > 0:
            avg_cost_long = np.average(long_term_chips_df['price'], weights=long_term_chips_df['percent'])
        else:
            avg_cost_long = prev_20d_close
        if not short_term_chips_df.empty and short_term_chips_df['percent'].sum() > 0:
            avg_cost_short = np.average(short_term_chips_df['price'], weights=short_term_chips_df['percent'])
        else:
            avg_cost_short = prev_20d_close
        return {
            'avg_cost_short_term': avg_cost_short,
            'avg_cost_long_term': avg_cost_long,
        }

    def _calculate_pressure_support(self) -> dict:
        close_price = self.ctx.get('close_price')
        if not close_price:
            return {}
        pressure_range_upper = close_price * 1.02
        support_range_lower = close_price * 0.98
        pressure_df = self.df[(self.df['price'] > close_price) & (self.df['price'] <= pressure_range_upper)]
        support_df = self.df[(self.df['price'] < close_price) & (self.df['price'] >= support_range_lower)]
        pressure_pct = pressure_df['percent'].sum()
        support_pct = support_df['percent'].sum()
        return {
            'pressure_above': pressure_pct,
            'support_below': support_pct,
            'pressure_above_volume': int(pressure_pct / 100 * self.ctx['total_chip_volume']),
            'support_below_volume': int(support_pct / 100 * self.ctx['total_chip_volume']),
        }

    def _calculate_effective_turnover(self) -> dict:
        low_price = self.ctx.get('low_price')
        high_price = self.ctx.get('high_price')
        cost_range_70_low, cost_range_70_high = self.ctx.get('cost_range_70pct', (None, None))
        if not all([low_price, high_price, cost_range_70_low, cost_range_70_high]):
            return {'turnover_volume_in_cost_range_70pct': None}
        intersection_low = max(low_price, cost_range_70_low)
        intersection_high = min(high_price, cost_range_70_high)
        daily_price_range = high_price - low_price
        intersection_range = intersection_high - intersection_low
        if daily_price_range > 0 and intersection_range > 0:
            effective_ratio = intersection_range / daily_price_range
            turnover_volume = int(self.ctx['daily_turnover_volume'] * effective_ratio)
        else:
            turnover_volume = 0
        return {'turnover_volume_in_cost_range_70pct': turnover_volume}

    def _calculate_advanced_structures(self, context: dict) -> dict:
        """【V7.0 · 裁汰版】移除基于估算的 peak_absorption_intensity"""
        results = {}
        peak_volume = context.get('peak_volume')
        total_float_share = self.ctx.get('total_chip_volume')
        if peak_volume is not None and total_float_share and total_float_share > 0:
            results['peak_control_ratio'] = (peak_volume / total_float_share) * 100
        close_price = self.ctx.get('close_price')
        if close_price:
            winners_df = self.df[self.df['price'] < close_price]
            if not winners_df.empty and winners_df['percent'].sum() > 0:
                winner_avg_cost = np.average(winners_df['price'], weights=winners_df['percent'])
                results['winner_avg_cost'] = winner_avg_cost
                if winner_avg_cost > 0:
                    results['winner_profit_margin'] = ((close_price - winner_avg_cost) / winner_avg_cost) * 100
                else:
                    results['winner_profit_margin'] = 0.0
            else:
                results['winner_avg_cost'] = None
                results['winner_profit_margin'] = 0.0
            losers_df = self.df[self.df['price'] > close_price]
            if not losers_df.empty and losers_df['percent'].sum() > 0:
                loser_avg_cost = np.average(losers_df['price'], weights=losers_df['percent'])
                results['loser_avg_cost'] = loser_avg_cost
            else:
                results['loser_avg_cost'] = None
        peak_cost = context.get('peak_cost') # 此行保留给 price_to_peak_ratio 使用
        if close_price and peak_cost and peak_cost > 0:
            results['price_to_peak_ratio'] = close_price / peak_cost
        if close_price:
            weighted_mean = self.ctx.get('weight_avg_cost')
            if weighted_mean is not None:
                weighted_std = np.sqrt(np.average((self.df['price'] - weighted_mean)**2, weights=self.df['percent']))
                if weighted_std and weighted_std > 0:
                    results['chip_zscore'] = (close_price - weighted_mean) / weighted_std
        return results

    def _calculate_turnover_microstructure(self, context: dict) -> dict:
        """【V2.0 · 裁汰版】精简指标，仅保留基于精确计算的 turnover_at_peak_ratio"""
        minute_df = self.ctx.get('minute_data')
        daily_turnover_vol = self.ctx.get('daily_turnover_volume')
        if minute_df is None or minute_df.empty or not daily_turnover_vol or daily_turnover_vol <= 0:
            return {'turnover_at_peak_ratio': None}
        total_vol_shares = minute_df['vol'].sum() * 100
        if total_vol_shares <= 0:
            return {'turnover_at_peak_ratio': 0.0}
        peak_range_low = context.get('peak_range_low')
        peak_range_high = context.get('peak_range_high')
        turnover_at_peak_ratio = None
        if peak_range_low is not None and peak_range_high is not None:
            vol_at_peak = minute_df[(minute_df['minute_vwap'] >= peak_range_low) & (minute_df['minute_vwap'] <= peak_range_high)]['vol'].sum() * 100
            turnover_at_peak_ratio = (vol_at_peak / total_vol_shares) * 100
        return {'turnover_at_peak_ratio': turnover_at_peak_ratio}

    def _calculate_concentration_dynamics(self, context: dict) -> dict:
        """【新增】计算筹码集中度四象限动态归因指标"""
        today_conc = context.get('concentration_90pct')
        prev_conc = self.ctx.get('prev_concentration_90pct')
        vol_above_ratio = self.ctx.get('volume_above_vwap_ratio')
        vol_below_ratio = self.ctx.get('volume_below_vwap_ratio')
        if not all(pd.notna(v) for v in [today_conc, prev_conc, vol_above_ratio, vol_below_ratio]):
            return {
                'concentration_increase_by_support': None, 'concentration_increase_by_chasing': None,
                'concentration_decrease_by_distribution': None, 'concentration_decrease_by_capitulation': None,
            }
        # 集中度数值越小越集中，所以delta_c为负代表集中度增加
        delta_c = today_conc - prev_conc
        # 集中度增加（delta_c < 0）
        increase_magnitude = abs(min(0, delta_c))
        increase_by_support = increase_magnitude * vol_below_ratio
        increase_by_chasing = increase_magnitude * vol_above_ratio
        # 集中度减少（delta_c > 0）
        decrease_magnitude = max(0, delta_c)
        decrease_by_distribution = decrease_magnitude * vol_above_ratio
        decrease_by_capitulation = decrease_magnitude * vol_below_ratio
        return {
            'concentration_increase_by_support': increase_by_support,
            'concentration_increase_by_chasing': increase_by_chasing,
            'concentration_decrease_by_distribution': decrease_by_distribution,
            'concentration_decrease_by_capitulation': decrease_by_capitulation,
        }
    
    def _get_turnover_in_range(self, low_bound, high_bound) -> float:
        """辅助函数：估算在某个价格区间的换手量"""
        low_price = self.ctx.get('low_price')
        high_price = self.ctx.get('high_price')
        daily_turnover = self.ctx.get('daily_turnover_volume')
        if not all([low_price, high_price, daily_turnover, daily_turnover > 0]):
            return 0
        intersection_low = max(low_price, low_bound)
        intersection_high = min(high_price, high_bound)
        daily_price_range = high_price - low_price
        intersection_range = intersection_high - intersection_low
        if daily_price_range > 0 and intersection_range > 0:
            effective_ratio = intersection_range / daily_price_range
            return daily_turnover * effective_ratio
        return 0

    def _calculate_chip_fault(self, context: dict) -> dict:
        """【V11.1 健壮性修复版】计算筹码断层指标。"""
        results = {
            'chip_fault_strength': None,
            'chip_fault_vacuum_percent': 0.0,
            'is_chip_fault_formed': False
        }
        peak_cost = context.get('peak_cost')
        close_price = self.ctx.get('close_price')
        if not all([peak_cost, close_price]):
            return results
        fault_strength = (close_price - peak_cost) / peak_cost if peak_cost > 0 else 0
        results['chip_fault_strength'] = fault_strength
        fault_zone_low = peak_cost * 1.01
        fault_zone_high = close_price * 0.99
        if fault_zone_high > fault_zone_low:
            fault_zone_df = self.df[(self.df['price'] >= fault_zone_low) & (self.df['price'] <= fault_zone_high)]
            vacuum_chip_percent = fault_zone_df['percent'].sum()
            results['chip_fault_vacuum_percent'] = vacuum_chip_percent
        else:
            results['chip_fault_vacuum_percent'] = 0.0
        is_strong_fault = fault_strength > 0.20
        vacuum_percent = results.get('chip_fault_vacuum_percent')
        if vacuum_percent is not None:
            is_vacuum_clear = vacuum_percent < 5.0
        else:
            is_vacuum_clear = False
        results['is_chip_fault_formed'] = is_strong_fault and is_vacuum_clear
        return results

    def _calculate_peak_dynamics(self, context: dict) -> dict:
        """【新增】计算主峰战役复盘指标"""
        minute_df = self.ctx.get('minute_data')
        peak_range_low = context.get('peak_range_low')
        peak_range_high = context.get('peak_range_high')
        peak_cost = context.get('peak_cost')
        total_daily_vol = self.ctx.get('daily_turnover_volume')
        results = {
            'peak_defense_intensity': None,
            'peak_vwap_deviation': None,
            'peak_net_volume_flow': None,
        }
        if minute_df is None or minute_df.empty or not all(pd.notna(v) for v in [peak_range_low, peak_range_high, peak_cost, total_daily_vol]) or total_daily_vol <= 0:
            return results
        peak_zone_df = minute_df[(minute_df['minute_vwap'] >= peak_range_low) & (minute_df['minute_vwap'] <= peak_range_high)].copy()
        if peak_zone_df.empty:
            results.update({'peak_defense_intensity': 0.0, 'peak_vwap_deviation': 0.0, 'peak_net_volume_flow': 0.0})
            return results
        vol_in_peak = peak_zone_df['vol'].sum() * 100
        if vol_in_peak > 0:
            results['peak_defense_intensity'] = (vol_in_peak / total_daily_vol) * 100
            amount_in_peak = (peak_zone_df['amount'] * 1000).sum()
            vwap_in_peak = amount_in_peak / vol_in_peak
            if peak_cost > 0:
                results['peak_vwap_deviation'] = (vwap_in_peak / peak_cost - 1) * 100
            if 'open' in peak_zone_df.columns and 'close' in peak_zone_df.columns:
                peak_zone_df['is_up_minute'] = peak_zone_df['close'] > peak_zone_df['open']
                peak_zone_df['is_down_minute'] = peak_zone_df['close'] < peak_zone_df['open']
                vol_up = peak_zone_df[peak_zone_df['is_up_minute']]['vol'].sum() * 100
                vol_down = peak_zone_df[peak_zone_df['is_down_minute']]['vol'].sum() * 100
                results['peak_net_volume_flow'] = (vol_up - vol_down) / vol_in_peak
        else:
            results.update({'peak_defense_intensity': 0.0, 'peak_vwap_deviation': 0.0, 'peak_net_volume_flow': 0.0})
        return results

    def _calculate_minute_derived_dynamics(self, context: dict) -> dict:
        """【新增】计算基于分钟K线的全面动态升维指标"""
        minute_df = self.ctx.get('minute_data')
        total_daily_vol = self.ctx.get('daily_turnover_volume')
        close_price = self.ctx.get('close_price')
        open_price = self.ctx.get('open_price') # 假设上下文中已有日开盘价
        results = {
            'realized_pressure_intensity': None,
            'realized_support_intensity': None,
            'profit_taking_urgency': None,
            'profit_realization_premium': None,
            'fault_breakthrough_intensity': None,
            'intraday_volume_gini': None,
            'volume_weighted_time_index': None,
            'intraday_trend_efficiency': None,
            'am_pm_vwap_ratio': None,
        }
        if minute_df is None or minute_df.empty or not total_daily_vol or total_daily_vol <= 0:
            return results
        # --- 1. 支撑与压力动态交火 ---
        if pd.notna(close_price):
            pressure_zone_low = close_price
            pressure_zone_high = close_price * 1.02
            support_zone_low = close_price * 0.98
            support_zone_high = close_price
            vol_in_pressure_zone = minute_df[(minute_df['minute_vwap'] >= pressure_zone_low) & (minute_df['minute_vwap'] <= pressure_zone_high)]['vol'].sum() * 100
            results['realized_pressure_intensity'] = (vol_in_pressure_zone / total_daily_vol) * 100
            vol_in_support_zone = minute_df[(minute_df['minute_vwap'] >= support_zone_low) & (minute_df['minute_vwap'] < support_zone_high)]['vol'].sum() * 100
            results['realized_support_intensity'] = (vol_in_support_zone / total_daily_vol) * 100
        # --- 2. 获利盘行为心理 ---
        prev_winner_avg_cost = self.ctx.get('prev_winner_avg_cost')
        if pd.notna(prev_winner_avg_cost) and prev_winner_avg_cost > 0:
            profit_taking_df = minute_df[minute_df['minute_vwap'] > prev_winner_avg_cost].copy()
            if not profit_taking_df.empty:
                vol_profit_taking = profit_taking_df['vol'].sum() * 100
                results['profit_taking_urgency'] = (vol_profit_taking / total_daily_vol) * 100
                amount_profit_taking = (profit_taking_df['amount'] * 1000).sum()
                vwap_profit_taking = amount_profit_taking / vol_profit_taking if vol_profit_taking > 0 else 0
                if vwap_profit_taking > 0:
                    results['profit_realization_premium'] = (vwap_profit_taking / prev_winner_avg_cost - 1) * 100
        # --- 3. 筹码断层突破动能 ---
        peak_cost = context.get('peak_cost')
        if pd.notna(peak_cost) and pd.notna(close_price) and close_price > peak_cost:
            fault_zone_low = peak_cost
            fault_zone_high = close_price
            vol_in_fault_zone = minute_df[(minute_df['minute_vwap'] >= fault_zone_low) & (minute_df['minute_vwap'] <= fault_zone_high)]['vol'].sum() * 100
            price_diff = close_price - peak_cost
            if vol_in_fault_zone > 0:
                results['fault_breakthrough_intensity'] = price_diff / vol_in_fault_zone
            elif price_diff > 0:
                results['fault_breakthrough_intensity'] = np.inf
        # --- 4. 成交量时间解构 ---
        volumes = minute_df['vol'].to_numpy()
        if volumes.sum() > 0:
            # Gini
            sorted_volumes = np.sort(volumes)
            n = len(volumes)
            cum_vol = np.cumsum(sorted_volumes, dtype=float)
            results['intraday_volume_gini'] = (n + 1 - 2 * np.sum(cum_vol) / cum_vol[-1]) / n
            # VWTI
            time_index = np.arange(1, n + 1)
            results['volume_weighted_time_index'] = np.sum(time_index * volumes) / (n * np.sum(volumes))
        # --- 5. 日内趋势质量 ---
        if pd.notna(open_price) and pd.notna(close_price):
            total_path = (minute_df['high'] - minute_df['low']).sum()
            net_change = close_price - open_price
            if total_path > 0:
                results['intraday_trend_efficiency'] = net_change / total_path
        # --- 6. 动态成本演化 ---
        minute_df['trade_time_obj'] = pd.to_datetime(minute_df['trade_time']).dt.time
        am_df = minute_df[minute_df['trade_time_obj'] < pd.to_datetime('12:00').time()]
        pm_df = minute_df[minute_df['trade_time_obj'] >= pd.to_datetime('13:00').time()]
        if not am_df.empty and not pm_df.empty:
            vwap_am = (am_df['amount'] * 1000).sum() / (am_df['vol'] * 100).sum() if (am_df['vol'] * 100).sum() > 0 else 0
            vwap_pm = (pm_df['amount'] * 1000).sum() / (pm_df['vol'] * 100).sum() if (pm_df['vol'] * 100).sum() > 0 else 0
            if vwap_am > 0 and vwap_pm > 0:
                results['am_pm_vwap_ratio'] = (vwap_pm / vwap_am - 1) * 100
        return results

    def _calculate_chip_interaction_dynamics(self, context: dict) -> dict:
        """【V2.0 · 海姆达尔之眼重构版】计算主力-散户筹码博弈矩阵"""
        minute_df = self.ctx.get('minute_data')
        total_daily_vol = self.ctx.get('daily_turnover_volume')
        close_price = self.ctx.get('close_price')
        results = {
            'main_force_suppressive_accumulation': None, 'retail_suppressive_accumulation': None,
            'main_force_rally_distribution': None, 'retail_rally_distribution': None,
            'main_force_capitulation_distribution': None, 'retail_capitulation_distribution': None,
            'main_force_chasing_accumulation': None, 'retail_chasing_accumulation': None,
            'main_force_t0_arbitrage': None, 'retail_t0_arbitrage': None,
        }
        # 检查分钟数据是否已被“奥丁之眼”算法增强
        required_cols = ['main_force_buy_vol', 'retail_sell_vol'] # 抽样检查关键列
        if minute_df is None or minute_df.empty or not total_daily_vol or total_daily_vol <= 0 or not pd.notna(close_price) or not all(c in minute_df.columns for c in required_cols):
            print(f"调试信息: 筹码交互计算跳过，因分钟数据不完整或未被资金流归因算法增强。")
            return results
        
        # 1. 定义筹码地图区域
        winners_df = self.df[self.df['price'] < close_price]
        losers_df = self.df[self.df['price'] > close_price]
        winner_zone = (winners_df['price'].min(), winners_df['price'].max()) if not winners_df.empty else (0, 0)
        loser_zone = (losers_df['price'].min(), losers_df['price'].max()) if not losers_df.empty else (np.inf, np.inf)
        # 2. 定义分钟级行为
        minute_df['is_up'] = minute_df['close'] > minute_df['open']
        minute_df['is_down'] = minute_df['close'] < minute_df['open']
        # 3. 识别每分钟的行为象限
        is_in_winner_zone = (minute_df['minute_vwap'] >= winner_zone[0]) & (minute_df['minute_vwap'] <= winner_zone[1])
        is_in_loser_zone = (minute_df['minute_vwap'] >= loser_zone[0]) & (minute_df['minute_vwap'] <= loser_zone[1])
        # 4. 计算8个核心博弈指标
        results['main_force_suppressive_accumulation'] = minute_df.loc[minute_df['is_up'] & is_in_loser_zone, 'main_force_buy_vol'].sum()
        results['retail_suppressive_accumulation'] = minute_df.loc[minute_df['is_up'] & is_in_loser_zone, 'retail_buy_vol'].sum()
        results['main_force_rally_distribution'] = minute_df.loc[minute_df['is_down'] & is_in_winner_zone, 'main_force_sell_vol'].sum()
        results['retail_rally_distribution'] = minute_df.loc[minute_df['is_down'] & is_in_winner_zone, 'retail_sell_vol'].sum()
        results['main_force_capitulation_distribution'] = minute_df.loc[minute_df['is_down'] & is_in_loser_zone, 'main_force_sell_vol'].sum()
        results['retail_capitulation_distribution'] = minute_df.loc[minute_df['is_down'] & is_in_loser_zone, 'retail_sell_vol'].sum()
        results['main_force_chasing_accumulation'] = minute_df.loc[minute_df['is_up'] & is_in_winner_zone, 'main_force_buy_vol'].sum()
        results['retail_chasing_accumulation'] = minute_df.loc[minute_df['is_up'] & is_in_winner_zone, 'retail_buy_vol'].sum()
        # 5. 计算T0套利
        cost_15pct = context.get('cost_15pct')
        cost_85pct = context.get('cost_85pct')
        if pd.notna(cost_15pct) and pd.notna(cost_85pct):
            mf_sell_high = minute_df.loc[minute_df['is_down'] & (minute_df['minute_vwap'] > cost_85pct), 'main_force_sell_vol'].sum()
            mf_buy_low = minute_df.loc[minute_df['is_up'] & (minute_df['minute_vwap'] < cost_15pct), 'main_force_buy_vol'].sum()
            results['main_force_t0_arbitrage'] = mf_sell_high + mf_buy_low
            retail_sell_high = minute_df.loc[minute_df['is_down'] & (minute_df['minute_vwap'] > cost_85pct), 'retail_sell_vol'].sum()
            retail_buy_low = minute_df.loc[minute_df['is_up'] & (minute_df['minute_vwap'] < cost_15pct), 'retail_buy_vol'].sum()
            results['retail_t0_arbitrage'] = retail_sell_high + retail_buy_low
        # 6. 归一化并填充结果
        for key in results:
            if results[key] is not None:
                results[key] = (results[key] / total_daily_vol) * 100
        return results

    def _calculate_cross_day_chip_flow(self, context: dict) -> dict:
        """【V1.5 - 诊断信息增强版】计算跨日筹码迁徙"""
        results = {
            'short_term_profit_taking_ratio': None,
            'long_term_chips_unlocked_ratio': None,
            'short_term_capitulation_ratio': None,
            'long_term_despair_selling_ratio': None,
        }
        prev_chips_df = context.get('prev_chip_distribution')
        prev_close = context.get('prev_close_price')
        prev_prev_20d_close = context.get('prev_prev_20d_close')
        daily_turnover_vol = context.get('daily_turnover_volume')
        is_data_invalid = False
        if prev_chips_df is None or prev_chips_df.empty:
            is_data_invalid = True
        for v in [prev_close, prev_prev_20d_close, daily_turnover_vol]:
            if v is None or pd.isnull(v):
                is_data_invalid = True
                break
        if is_data_invalid or daily_turnover_vol <= 0:
            # 增强调试信息，明确指出问题发生的股票和日期
            stock_code = context.get('stock_code', 'UNKNOWN_STOCK')
            trade_date = context.get('trade_date', 'UNKNOWN_DATE')
            print(f"调试信息: [{stock_code}] 在 [{trade_date}] 跨日筹码流计算跳过，因T-1日数据不完整。")
            
            return results
        prev_winners = prev_chips_df[prev_chips_df['price'] < prev_close]
        prev_losers = prev_chips_df[prev_chips_df['price'] > prev_close]
        st_winners_pct = prev_winners[prev_winners['price'] >= prev_prev_20d_close]['percent'].sum()
        lt_winners_pct = prev_winners[prev_winners['price'] < prev_prev_20d_close]['percent'].sum()
        st_losers_pct = prev_losers[prev_losers['price'] >= prev_prev_20d_close]['percent'].sum()
        lt_losers_pct = prev_losers[prev_losers['price'] < prev_prev_20d_close]['percent'].sum()
        results['short_term_profit_taking_ratio'] = st_winners_pct
        results['long_term_chips_unlocked_ratio'] = lt_winners_pct
        results['short_term_capitulation_ratio'] = st_losers_pct
        results['long_term_despair_selling_ratio'] = lt_losers_pct
        return results


















