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
        """【V22.5 · 静默行军版】根据上下文标记，实现条件性日志输出。"""
        # [代码新增开始] 检查是否需要显示探针日志
        show_probes = self.ctx.get('is_last_day_in_batch', False)
        if show_probes:
            self._probe_chip_calculation_readiness()
        # [代码新增结束]
        if self.df.empty or not all(k in self.ctx for k in ['weight_avg', 'winner_rate', 'cost_95pct', 'cost_5pct', 'close_price', 'total_chip_volume']):
            # [代码修改开始] 即使计算终止，也应在最后一天显示探针信息
            if show_probes:
                print(f"--- [计算引擎] [{self.ctx.get('stock_code')}] 前置检查失败，核心上下文缺失，计算终止。")
            # [代码修改结束]
            return {}
        # [代码修改开始] 将所有日志打印置于条件块内
        if show_probes:
            print(f"--- [计算引擎] [{self.ctx.get('stock_code')}] 开始计算 {self.ctx.get('trade_date')} 的指标 ---")
        summary_info = self._get_summary_metrics_from_context()
        if show_probes:
            print(f"  [探针] _get_summary_metrics_from_context -> total_winner_rate: {summary_info.get('total_winner_rate')}")
        self.ctx.update(summary_info)
        peaks_info = self._calculate_peaks()
        if show_probes:
            print(f"  [探针] _calculate_peaks -> peak_cost: {peaks_info.get('peak_cost')}, peak_volume: {peaks_info.get('peak_volume')}")
        concentration_info = self._calculate_concentration_from_perf()
        if show_probes:
            print(f"  [探针] _calculate_concentration_from_perf -> concentration_90pct: {concentration_info.get('concentration_90pct')}")
        winner_structure_info = self._calculate_winner_structure()
        if show_probes:
            print(f"  [探针] _calculate_winner_structure -> winner_rate_short_term: {winner_structure_info.get('winner_rate_short_term')}")
        holder_costs_info = self._calculate_holder_costs()
        if show_probes:
            print(f"  [探针] _calculate_holder_costs -> avg_cost_short_term: {holder_costs_info.get('avg_cost_short_term')}")
        pressure_support_info = self._calculate_pressure_support()
        if show_probes:
            print(f"  [探针] _calculate_pressure_support -> pressure_above: {pressure_support_info.get('pressure_above')}")
        context_for_derived_metrics = {
            **self.ctx,
            **peaks_info,
            **concentration_info,
            **winner_structure_info,
            **holder_costs_info,
        }
        advanced_structure_info = self._calculate_advanced_structures(context_for_derived_metrics)
        if show_probes:
            print(f"  [探针] _calculate_advanced_structures -> winner_avg_cost: {advanced_structure_info.get('winner_avg_cost')}")
        context_for_derived_metrics.update(advanced_structure_info)
        minute_derived_dynamics_info = self._calculate_minute_derived_dynamics(context_for_derived_metrics)
        if show_probes:
            print(f"  [探针] _calculate_minute_derived_dynamics -> profit_taking_urgency: {minute_derived_dynamics_info.get('profit_taking_urgency')}")
        chip_interaction_info = self._calculate_chip_interaction_dynamics(context_for_derived_metrics)
        if show_probes:
            print(f"  [探针] _calculate_chip_interaction_dynamics -> main_force_suppressive_accumulation: {chip_interaction_info.get('main_force_suppressive_accumulation')}")
        cross_day_flow_info = self._calculate_cross_day_chip_flow(context_for_derived_metrics)
        if show_probes:
            print(f"  [探针] _calculate_cross_day_chip_flow -> short_term_profit_taking_ratio: {cross_day_flow_info.get('short_term_profit_taking_ratio')}")
        turnover_microstructure_info = self._calculate_turnover_microstructure(context_for_derived_metrics)
        concentration_dynamics_info = self._calculate_concentration_dynamics(context_for_derived_metrics)
        peak_dynamics_info = self._calculate_peak_dynamics(context_for_derived_metrics)
        fault_info = self._calculate_chip_fault(context_for_derived_metrics)
        cost_divergence_info = self._calculate_cost_divergence(context_for_derived_metrics)
        context_for_derived_metrics.update(cost_divergence_info)
        health_score_info = self._calculate_health_score(context_for_derived_metrics)
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
            **cross_day_flow_info,
            **advanced_structure_info,
            **fault_info,
            **cost_divergence_info,
            **health_score_info
        }
        if 'total_winner_rate' in summary_info:
            all_metrics['total_winner_rate'] = summary_info['total_winner_rate']
        if 'total_loser_rate' in winner_structure_info:
            all_metrics['total_loser_rate'] = winner_structure_info['total_loser_rate']
        all_metrics.pop('peak_range_low', None)
        all_metrics.pop('peak_range_high', None)
        if show_probes:
            self._probe_final_metrics(all_metrics)
        # [代码修改结束]
        return all_metrics

    def _prepare_minute_data_features(self):
        """【V2.1 · API单位对齐版】对单日分钟数据进行类型降级，减少内存占用。"""
        minute_df = self.ctx.get('minute_data')
        if minute_df is None or minute_df.empty:
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
        # [代码修改开始] 根据API文档，分钟线的amount单位是元，vol单位是股，无需转换。
        total_amount_yuan = minute_df['amount'].sum()
        total_vol_shares = minute_df['vol'].sum()
        # [代码修改结束]
        daily_vwap = total_amount_yuan / total_vol_shares if total_vol_shares > 0 else None
        self.ctx['daily_vwap'] = daily_vwap
        if daily_vwap is None:
            self.ctx.update({'volume_above_vwap_ratio': None, 'volume_below_vwap_ratio': None})
            return
        # [代码修改开始] 根据API文档，分钟线的amount单位是元，vol单位是股，无需转换。
        minute_df['minute_vwap'] = (minute_df['amount'] / minute_df['vol'].replace(0, np.nan)).astype('float32', errors='ignore')
        # [代码修改结束]
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
        """【V13.7 · 定义统一化修正版】使用 prev_20d_close 统一划分长短期筹码"""
        close_price = self.ctx.get('close_price')
        # [代码修改开始] 引入正确的长短期划分基准
        prev_20d_close = self.ctx.get('prev_20d_close')
        total_winner_rate = self.ctx.get('total_winner_rate', 0.0)
        # 检查所有必需的上下文变量
        if not all(pd.notna(v) for v in [close_price, prev_20d_close]):
            return {
                'winner_rate_short_term': None, 'winner_rate_long_term': None,
                'loser_rate_short_term': None, 'loser_rate_long_term': None,
                'total_loser_rate': 100.0 - total_winner_rate if total_winner_rate is not None else None
            }
        winners_df = self.df[self.df['price'] < close_price]
        losers_df = self.df[self.df['price'] > close_price]
        # 使用 prev_20d_close 作为长短期的分界线，与 _calculate_holder_costs 保持一致
        long_term_winner_rate = winners_df[winners_df['price'] < prev_20d_close]['percent'].sum()
        short_term_winner_rate = winners_df[winners_df['price'] >= prev_20d_close]['percent'].sum()
        long_term_loser_rate = losers_df[losers_df['price'] < prev_20d_close]['percent'].sum()
        short_term_loser_rate = losers_df[losers_df['price'] >= prev_20d_close]['percent'].sum()
        # [代码修改结束]
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
        """【V2.1 · API单位对齐版】精简指标，仅保留基于精确计算的 turnover_at_peak_ratio"""
        minute_df = self.ctx.get('minute_data')
        daily_turnover_vol = self.ctx.get('daily_turnover_volume')
        if minute_df is None or minute_df.empty or not daily_turnover_vol or daily_turnover_vol <= 0:
            return {'turnover_at_peak_ratio': None}
        # [代码修改开始] 根据API文档，分钟线的vol单位是股，无需转换。
        total_vol_shares = minute_df['vol'].sum()
        if total_vol_shares <= 0:
            return {'turnover_at_peak_ratio': 0.0}
        peak_range_low = context.get('peak_range_low')
        peak_range_high = context.get('peak_range_high')
        turnover_at_peak_ratio = None
        if peak_range_low is not None and peak_range_high is not None:
            vol_at_peak = minute_df[(minute_df['minute_vwap'] >= peak_range_low) & (minute_df['minute_vwap'] <= peak_range_high)]['vol'].sum()
            turnover_at_peak_ratio = (vol_at_peak / total_vol_shares) * 100
        # [代码修改结束]
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
        """【V1.2 · API单位对齐版】计算主峰战役复盘指标"""
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
            return results
        # [代码修改开始] 根据API文档，分钟线的vol单位是股，amount单位是元，无需转换。
        vol_in_peak = peak_zone_df['vol'].sum()
        if vol_in_peak > 0:
            results['peak_defense_intensity'] = (vol_in_peak / total_daily_vol) * 100
            amount_in_peak = peak_zone_df['amount'].sum()
            vwap_in_peak = amount_in_peak / vol_in_peak
            if peak_cost > 0:
                results['peak_vwap_deviation'] = (vwap_in_peak / peak_cost - 1) * 100
            if 'open' in peak_zone_df.columns and 'close' in peak_zone_df.columns:
                peak_zone_df['is_up_minute'] = peak_zone_df['close'] > peak_zone_df['open']
                peak_zone_df['is_down_minute'] = peak_zone_df['close'] < peak_zone_df['open']
                vol_up = peak_zone_df[peak_zone_df['is_up_minute']]['vol'].sum()
                vol_down = peak_zone_df[peak_zone_df['is_down_minute']]['vol'].sum()
                results['peak_net_volume_flow'] = (vol_up - vol_down) / vol_in_peak
        # [代码修改结束]
        return results

    def _calculate_minute_derived_dynamics(self, context: dict) -> dict:
        """【V1.3 · 全天候协议版】增强对半日交易等极端情况的处理"""
        minute_df = self.ctx.get('minute_data')
        total_daily_vol = self.ctx.get('daily_turnover_volume')
        close_price = self.ctx.get('close_price')
        open_price = self.ctx.get('open_price')
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
        if pd.notna(close_price):
            pressure_zone_low = close_price
            pressure_zone_high = close_price * 1.02
            support_zone_low = close_price * 0.98
            support_zone_high = close_price
            vol_in_pressure_zone = minute_df[(minute_df['minute_vwap'] >= pressure_zone_low) & (minute_df['minute_vwap'] <= pressure_zone_high)]['vol'].sum()
            results['realized_pressure_intensity'] = (vol_in_pressure_zone / total_daily_vol) * 100
            vol_in_support_zone = minute_df[(minute_df['minute_vwap'] >= support_zone_low) & (minute_df['minute_vwap'] < support_zone_high)]['vol'].sum()
            results['realized_support_intensity'] = (vol_in_support_zone / total_daily_vol) * 100
        prev_winner_avg_cost = self.ctx.get('prev_winner_avg_cost')
        if pd.notna(prev_winner_avg_cost) and prev_winner_avg_cost > 0:
            profit_taking_df = minute_df[minute_df['minute_vwap'] > prev_winner_avg_cost].copy()
            if not profit_taking_df.empty:
                vol_profit_taking = profit_taking_df['vol'].sum()
                results['profit_taking_urgency'] = (vol_profit_taking / total_daily_vol) * 100
                amount_profit_taking = profit_taking_df['amount'].sum()
                vwap_profit_taking = amount_profit_taking / vol_profit_taking if vol_profit_taking > 0 else 0
                if vwap_profit_taking > 0:
                    results['profit_realization_premium'] = (vwap_profit_taking / prev_winner_avg_cost - 1) * 100
        peak_cost = context.get('peak_cost')
        if pd.notna(peak_cost) and pd.notna(close_price) and close_price > peak_cost:
            fault_zone_low = peak_cost
            fault_zone_high = close_price
            vol_in_fault_zone = minute_df[(minute_df['minute_vwap'] >= fault_zone_low) & (minute_df['minute_vwap'] <= fault_zone_high)]['vol'].sum()
            price_diff = close_price - peak_cost
            if vol_in_fault_zone > 0:
                results['fault_breakthrough_intensity'] = price_diff / vol_in_fault_zone
            elif price_diff > 0:
                results['fault_breakthrough_intensity'] = price_diff * 1e12
        volumes = minute_df['vol'].to_numpy()
        if volumes.sum() > 0:
            sorted_volumes = np.sort(volumes)
            n = len(volumes)
            cum_vol = np.cumsum(sorted_volumes, dtype=float)
            results['intraday_volume_gini'] = (n + 1 - 2 * np.sum(cum_vol) / cum_vol[-1]) / n
            time_index = np.arange(1, n + 1)
            results['volume_weighted_time_index'] = np.sum(time_index * volumes) / (n * np.sum(volumes))
        if pd.notna(open_price) and pd.notna(close_price):
            total_path = (minute_df['high'] - minute_df['low']).sum()
            net_change = abs(close_price - open_price) # [代码修改] 使用绝对值来衡量总位移
            if total_path > 0:
                results['intraday_trend_efficiency'] = (close_price - open_price) / total_path # 保持方向性
            elif net_change == 0: # [代码修改] 如果路径和位移都为0，则效率为0
                results['intraday_trend_efficiency'] = 0.0
        minute_df['trade_time_obj'] = pd.to_datetime(minute_df['trade_time']).dt.time
        am_df = minute_df[minute_df['trade_time_obj'] < pd.to_datetime('12:00').time()]
        pm_df = minute_df[minute_df['trade_time_obj'] >= pd.to_datetime('13:00').time()]
        # [代码修改开始] 增强对半日交易等情况的处理
        am_vol_sum = am_df['vol'].sum()
        pm_vol_sum = pm_df['vol'].sum()
        vwap_am = am_df['amount'].sum() / am_vol_sum if am_vol_sum > 0 else 0
        vwap_pm = pm_df['amount'].sum() / pm_vol_sum if pm_vol_sum > 0 else 0
        if vwap_am > 0 and vwap_pm > 0:
            results['am_pm_vwap_ratio'] = (vwap_pm / vwap_am - 1) * 100
        elif vwap_am == 0 and vwap_pm > 0: # 仅下午交易
            results['am_pm_vwap_ratio'] = 100.0 # 定义为极大值
        elif vwap_am > 0 and vwap_pm == 0: # 仅上午交易
            results['am_pm_vwap_ratio'] = -100.0 # 定义为极小值
        else: # 全天无成交
            results['am_pm_vwap_ratio'] = 0.0
        # [代码修改结束]
        return results

    def _calculate_chip_interaction_dynamics(self, context: dict) -> dict:
        """【V2.2 · 健壮区域划分版】计算主力-散户筹码博弈矩阵"""
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
        required_cols = ['main_force_buy_vol', 'retail_sell_vol']
        if minute_df is None or minute_df.empty or not total_daily_vol or total_daily_vol <= 0 or not pd.notna(close_price) or not all(c in minute_df.columns for c in required_cols):
            return results
        # [代码修改开始] 增加对获利盘/套牢盘区域是否存在的健壮性检查
        winners_df = self.df[self.df['price'] < close_price]
        losers_df = self.df[self.df['price'] > close_price]
        winner_zone_exists = not winners_df.empty
        loser_zone_exists = not losers_df.empty
        winner_zone = (winners_df['price'].min(), winners_df['price'].max()) if winner_zone_exists else (0, 0)
        loser_zone = (losers_df['price'].min(), losers_df['price'].max()) if loser_zone_exists else (close_price, np.inf)
        # [代码修改结束]
        minute_df['is_up'] = minute_df['close'] > minute_df['open']
        minute_df['is_down'] = minute_df['close'] < minute_df['open']
        # [代码修改开始] 仅在区域存在时进行计算
        is_in_winner_zone = (minute_df['minute_vwap'] >= winner_zone[0]) & (minute_df['minute_vwap'] <= winner_zone[1]) if winner_zone_exists else pd.Series(False, index=minute_df.index)
        is_in_loser_zone = (minute_df['minute_vwap'] >= loser_zone[0]) & (minute_df['minute_vwap'] <= loser_zone[1]) if loser_zone_exists else pd.Series(False, index=minute_df.index)
        # [代码修改结束]
        results['main_force_suppressive_accumulation'] = minute_df.loc[minute_df['is_up'] & is_in_loser_zone, 'main_force_buy_vol'].sum()
        results['retail_suppressive_accumulation'] = minute_df.loc[minute_df['is_up'] & is_in_loser_zone, 'retail_buy_vol'].sum()
        results['main_force_rally_distribution'] = minute_df.loc[minute_df['is_down'] & is_in_winner_zone, 'main_force_sell_vol'].sum()
        results['retail_rally_distribution'] = minute_df.loc[minute_df['is_down'] & is_in_winner_zone, 'retail_sell_vol'].sum()
        results['main_force_capitulation_distribution'] = minute_df.loc[minute_df['is_down'] & is_in_loser_zone, 'main_force_sell_vol'].sum()
        results['retail_capitulation_distribution'] = minute_df.loc[minute_df['is_down'] & is_in_loser_zone, 'retail_sell_vol'].sum()
        results['main_force_chasing_accumulation'] = minute_df.loc[minute_df['is_up'] & is_in_winner_zone, 'main_force_buy_vol'].sum()
        results['retail_chasing_accumulation'] = minute_df.loc[minute_df['is_up'] & is_in_winner_zone, 'retail_buy_vol'].sum()
        cost_15pct = context.get('cost_15pct')
        cost_85pct = context.get('cost_85pct')
        if pd.notna(cost_15pct) and pd.notna(cost_85pct):
            mf_sell_high = minute_df.loc[minute_df['is_down'] & (minute_df['minute_vwap'] > cost_85pct), 'main_force_sell_vol'].sum()
            mf_buy_low = minute_df.loc[minute_df['is_up'] & (minute_df['minute_vwap'] < cost_15pct), 'main_force_buy_vol'].sum()
            results['main_force_t0_arbitrage'] = mf_sell_high + mf_buy_low
            retail_sell_high = minute_df.loc[minute_df['is_down'] & (minute_df['minute_vwap'] > cost_85pct), 'retail_sell_vol'].sum()
            retail_buy_low = minute_df.loc[minute_df['is_up'] & (minute_df['minute_vwap'] < cost_15pct), 'retail_buy_vol'].sum()
            results['retail_t0_arbitrage'] = retail_sell_high + retail_buy_low
        for key in results:
            if results[key] is not None:
                results[key] = (results[key] / total_daily_vol) * 100
        return results

    def _calculate_cross_day_chip_flow(self, context: dict) -> dict:
        """【V2.2 · 终极变量引用修正版】计算跨日筹码迁徙"""
        results = {
            'short_term_profit_taking_ratio': None,
            'long_term_chips_unlocked_ratio': None,
            'short_term_capitulation_ratio': None,
            'long_term_despair_selling_ratio': None,
        }
        prev_chips_df = context.get('prev_chip_distribution')
        prev_close = context.get('prev_close_price')
        prev_20d_close = context.get('prev_20d_close')
        daily_turnover_vol = context.get('daily_turnover_volume')
        missing_keys = []
        if prev_chips_df is None or prev_chips_df.empty:
            missing_keys.append('prev_chip_distribution')
        if prev_close is None or pd.isnull(prev_close):
            missing_keys.append('prev_close_price')
        if prev_20d_close is None or pd.isnull(prev_20d_close):
            missing_keys.append('prev_20d_close')
        if daily_turnover_vol is None or pd.isnull(daily_turnover_vol) or daily_turnover_vol <= 0:
            missing_keys.append('daily_turnover_volume')
        if missing_keys:
            is_first_day = context.get('is_first_day_in_batch', False)
            if not is_first_day and self.ctx.get('is_last_day_in_batch', False):
                stock_code = context.get('stock_code', 'UNKNOWN_STOCK')
                trade_date = context.get('trade_date', 'UNKNOWN_DATE')
            return results
        prev_winners = prev_chips_df[prev_chips_df['price'] < prev_close]
        prev_losers = prev_chips_df[prev_chips_df['price'] > prev_close]
        st_winners_pct = prev_winners[prev_winners['price'] >= prev_20d_close]['percent'].sum()
        lt_winners_pct = prev_winners[prev_winners['price'] < prev_20d_close]['percent'].sum()
        st_losers_pct = prev_losers[prev_losers['price'] >= prev_20d_close]['percent'].sum()
        # [代码修改开始] 修正变量引用错误，将 losers_df 修正为 prev_losers
        lt_losers_pct = prev_losers[prev_losers['price'] < prev_20d_close]['percent'].sum()
        # [代码修改结束]
        results['short_term_profit_taking_ratio'] = st_winners_pct
        results['long_term_chips_unlocked_ratio'] = lt_winners_pct
        results['short_term_capitulation_ratio'] = st_losers_pct
        results['long_term_despair_selling_ratio'] = lt_losers_pct
        return results

    def _calculate_cost_divergence(self, context: dict) -> dict:
        """【新增】计算成本乖离率"""
        avg_cost_short = context.get('avg_cost_short_term')
        avg_cost_long = context.get('avg_cost_long_term')
        if avg_cost_short is None or avg_cost_long is None or avg_cost_long == 0:
            return {'cost_divergence': None}
        divergence = (avg_cost_short / avg_cost_long - 1) * 100
        return {'cost_divergence': divergence}

    def _calculate_health_score(self, context: dict) -> dict:
        """【新增】计算筹码健康分"""
        # 定义健康分构成要素及其权重
        # 权重为负表示该指标值越小越健康
        components = {
            'concentration_90pct': -0.25,
            'winner_profit_margin': 0.30,
            'cost_divergence': -0.20,
            'peak_stability': 0.25,
        }
        # 归一化范围
        SCORE_MIN, SCORE_MAX = 0, 100
        # 预设的各指标经验上的“差”值和“优”值，用于归一化
        # concentration_90pct: 50%为极差, 5%为极优
        # winner_profit_margin: -20%为极差, 50%为极优
        # cost_divergence: 15%为极差, -5%为极优
        # peak_stability: 1为极差, 20为极优
        normalization_ranges = {
            'concentration_90pct': (0.50, 0.05),
            'winner_profit_margin': (-0.20, 0.50),
            'cost_divergence': (0.15, -0.05),
            'peak_stability': (1.0, 20.0),
        }
        total_score = 0
        total_weight = 0
        for metric, weight in components.items():
            value = context.get(metric)
            if value is None or not pd.notna(value):
                continue # 如果某个组件缺失，则跳过
            min_val, max_val = normalization_ranges[metric]
            # 归一化处理
            # (value - min) / (max - min)
            normalized_value = (value - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else 0.5
            # 裁剪到 [0, 1] 范围
            clipped_value = np.clip(normalized_value, 0, 1)
            # 映射到 [0, 100] 分数
            score = clipped_value * (SCORE_MAX - SCORE_MIN) + SCORE_MIN
            total_score += score * abs(weight)
            total_weight += abs(weight)
        if total_weight == 0:
            return {'chip_health_score': None}
        # 根据权重调整最终分数
        final_score = total_score / total_weight
        return {'chip_health_score': final_score}



    def _probe_chip_calculation_readiness(self):
        """【V-Probe 2.1 · 静默行军版】根据上下文标记，实现条件性日志输出。"""
        # [代码新增开始] 增加条件判断，仅在最后一天执行
        if not self.ctx.get('is_last_day_in_batch', False):
            return
        # [代码新增结束]
        print("\n" + "="*20 + " 筹码计算战备状态探针 " + "="*20)
        stock_code = self.ctx.get('stock_code', 'UNKNOWN')
        trade_date = self.ctx.get('trade_date', 'UNKNOWN')
        print(f"探针目标: [{stock_code}] 日期: [{trade_date}]")
        is_ready = True
        minute_df = self.ctx.get('minute_data')
        required_cols = ['main_force_buy_vol', 'retail_sell_vol']
        if minute_df is None or minute_df.empty or not all(c in minute_df.columns for c in required_cols):
            print("  - [依赖链诊断 - 失败]: 分钟数据未被资金流归因算法增强。")
            print("    原因: 很可能当天的日线资金流数据(fund_flow)缺失，导致增强步骤跳过。")
            print("    影响: 所有【主力-散户筹码博弈】指标将无法计算 (例如: main_force_suppressive_accumulation)。")
            is_ready = False
        else:
            print("  - [依赖链诊断 - 成功]: 分钟数据已成功增强。")
        prev_chip_dist = self.ctx.get('prev_chip_distribution')
        if prev_chip_dist is None or prev_chip_dist.empty:
            is_first_day = self.ctx.get('is_first_day_in_batch', False)
            if not is_first_day:
                print("  - [记忆链诊断 - 失败]: T-1日上下文数据(prev_chip_distribution)缺失。")
                print("    原因: 跨区块的'记忆'未能成功传递。")
                print("    影响: 所有【跨日筹码迁徙】指标将无法计算 (例如: short_term_profit_taking_ratio)。")
                is_ready = False
            else:
                print("  - [记忆链诊断 - 正常]: 区块首日，无T-1日上下文数据。")
        else:
            print("  - [记忆链诊断 - 成功]: T-1日上下文数据已就绪。")
        print("--- [上下文深度探针] ---")
        context_keys_to_probe = ['prev_concentration_90pct', 'prev_winner_avg_cost', 'prev_day_20d_ago_close']
        for key in context_keys_to_probe:
            value = self.ctx.get(key)
            status = f"值: {value:.4f}" if isinstance(value, (int, float)) and pd.notna(value) else f"状态: {value}"
            print(f"  >>> T-1日上下文 '{key}': {status}")
        if is_ready:
            print("探针结论: 所有关键依赖项均已就绪，计算可以全面展开。")
        else:
            print("探针结论: 存在关键依赖项缺失，部分高级指标将为空。")
        print("="*20 + " 探针诊断结束 " + "="*20 + "\n")

    def _probe_final_metrics(self, metrics: dict):
        """【V2.1 · 静默行军版】根据上下文标记，实现条件性日志输出。"""
        # [代码新增开始] 增加条件判断，仅在最后一天执行
        if not self.ctx.get('is_last_day_in_batch', False):
            return
        # [代码新增结束]
        print("--- [最终指标审查探针] ---")
        critical_metrics = [
            'peak_cost', 'concentration_90pct', 'winner_avg_cost',
            'short_term_profit_taking_ratio', 'main_force_suppressive_accumulation',
            'profit_taking_urgency', 'cost_divergence', 'chip_health_score'
        ]
        has_issues = False
        for metric_name in critical_metrics:
            value = metrics.get(metric_name)
            if value is None:
                print(f"  >>> 警告: 关键指标 '{metric_name}' 计算结果为 None。")
                has_issues = True
            elif isinstance(value, (int, float)) and value == 0:
                if metric_name not in ['main_force_suppressive_accumulation']:
                    print(f"  >>> 注意: 关键指标 '{metric_name}' 计算结果为 0。请确认是否符合预期。")
        if not has_issues:
            print("  >>> 审查通过: 所有受监控的关键指标均已成功计算出有效值。")
        print("--- [最终审查结束] ---")









