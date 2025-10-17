# 文件: services/chip_feature_calculator.py

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from decimal import Decimal

class ChipFeatureCalculator:
    """
    【V10.0 最终正确版】
    - 根本性修复: 废除了所有历史版本中错误的获利盘计算逻辑。
    - 回归定义: _calculate_winner_structure 方法现在严格按照模型字段的
                业务定义进行计算，确保在接收到真实、正确的收盘价后，
                能够准确反映市场的获利盘结构。
    - 功能增强: 新增了 'total_winner_rate' (总获利盘) 指标的计算。
    """
    def __init__(self, daily_chips_df: pd.DataFrame, context_data: dict):
        # 初始化函数：将 Decimal 类型转换为 float，便于计算
        self.df = daily_chips_df.reset_index(drop=True)
        self.ctx = context_data
        
        if not self.df.empty:
            percent_sum = self.df['percent'].sum()
            # 检查总和是否接近100，且大于0，以避免除以0的错误
            if not np.isclose(percent_sum, 100.0) and percent_sum > 0:
                # print(f"    -> [数据清洗] 注意：检测到筹码分布总和为 {percent_sum:.2f}，不等于100。正在执行强制归一化...")
                self.df['percent'] = (self.df['percent'] / percent_sum) * 100.0
        
        cyq_perf_fields = [
            'total_chip_volume', 'daily_turnover_volume', 'close_price', 'high_price', 'low_price', 'prev_20d_close',
            'cost_5pct', 'cost_15pct', 'cost_50pct', 'cost_85pct', 'cost_95pct', 'weight_avg', 'winner_rate'
        ]
        for key in cyq_perf_fields:
            if key in self.ctx and isinstance(self.ctx[key], Decimal):
                self.ctx[key] = float(self.ctx[key])

    def calculate_all_metrics(self) -> dict:
        """
        【V13.0 cyq_perf 数据源升维版】
        - 调用重构后的 _get_summary_metrics_from_context 和 _calculate_concentration_from_perf 方法。
        """
        # --- 0. 前置检查 ---
        # 检查 cyq_perf 提供的关键字段
        if self.df.empty or not all(k in self.ctx for k in ['weight_avg', 'winner_rate', 'cost_95pct', 'cost_5pct', 'close_price', 'total_chip_volume']):
            return {}

        # --- 1. 基础指标计算 ---
        # 调用新的方法
        summary_info = self._get_summary_metrics_from_context()
        self.ctx.update(summary_info) # 必须先更新上下文，后续计算会用到
        peaks_info = self._calculate_peaks()
        concentration_info = self._calculate_concentration_from_perf() # 调用新方法
        winner_structure_info = self._calculate_winner_structure()
        holder_costs_info = self._calculate_holder_costs()
        pressure_support_info = self._calculate_pressure_support()
        turnover_info = self._calculate_effective_turnover()
        turnover_structure_info = self._calculate_turnover_structure()
        turnover_at_peak_info = self._calculate_turnover_at_peak(peaks_info)

        # --- 2. 构建升维计算所需的“增强上下文” ---
        context_for_derived_metrics = {
            **self.ctx, 
            **peaks_info, 
            **concentration_info, 
            **winner_structure_info,
            **holder_costs_info,
        }

        # --- 3. 升维指标计算 ---
        advanced_structure_info = self._calculate_advanced_structures(context_for_derived_metrics)
        fault_info = self._calculate_chip_fault(context_for_derived_metrics)

        # --- 4. 合并所有结果并返回 ---
        all_metrics = {
            **peaks_info,
            **concentration_info,
            **winner_structure_info,
            **holder_costs_info,
            **pressure_support_info,
            **turnover_info,
            **turnover_structure_info,
            **turnover_at_peak_info,
            **advanced_structure_info,
            **fault_info
        }
        if 'total_winner_rate' in summary_info:
            all_metrics['total_winner_rate'] = summary_info['total_winner_rate']
        if 'total_loser_rate' in winner_structure_info:
            all_metrics['total_loser_rate'] = winner_structure_info['total_loser_rate']

        all_metrics.pop('peak_range_low', None)
        all_metrics.pop('peak_range_high', None)

        # --- 5. 返回清理后的最终结果 ---
        return all_metrics

    def _get_summary_metrics_from_context(self) -> dict:
        """
        【V13.0 重构】直接从上下文中提取由 cyq_perf 提供的高阶指标。
        - 替代了原有的 _calculate_summary_metrics 方法。
        """
        # 直接从 self.ctx 获取由 cyq_perf 提供的、更准确的指标
        weight_avg_cost = self.ctx.get('weight_avg')
        total_winner_rate = self.ctx.get('winner_rate')

        return {
            'weight_avg_cost': weight_avg_cost,
            'total_winner_rate': total_winner_rate,
        }

    def _calculate_peaks(self) -> dict:
        """
        【V10.1 逻辑统一版】
        - 优化: 统一了单峰和多峰情况下，主峰范围(peak_range)的计算逻辑，
                使其全部基于 find_peaks 的返回结果，消除了硬编码。
        """
        # : 增加 width 参数，让 find_peaks 计算山峰宽度
        peaks, properties = find_peaks(self.df['percent'], prominence=0.1, width=1)
        
        # 辅助函数，用于从插值索引中获取价格
        def get_price_from_interpolated_index(interp_idx):
            price_indices = self.df.index.to_numpy()
            price_values = self.df['price'].to_numpy()
            return np.interp(interp_idx, price_indices, price_values)

        if len(peaks) == 0:
            # 如果 find_peaks 未找到任何山峰，则使用全局最大值作为唯一山峰
            peak_idx = self.df['percent'].idxmax()
            peak_price = self.df.loc[peak_idx, 'price']
            # : 单峰情况下，也提供一个默认的、较窄的范围，避免后续计算出错
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

        # ▼▼▼ 统一使用插值法获取精确边界 ▼▼▼
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
        """
        【V13.0 重构】直接基于 cyq_perf 提供的成本分位数计算筹码集中度。
        - 替代了原有的 _calculate_concentration 方法。
        - 效率和准确性都得到极大提升。
        """
        # 从上下文中获取 cyq_perf 提供的成本分位数
        cost_95pct = self.ctx.get('cost_95pct')
        cost_5pct = self.ctx.get('cost_5pct')
        cost_85pct = self.ctx.get('cost_85pct')
        cost_15pct = self.ctx.get('cost_15pct')
        weight_avg_cost = self.ctx.get('weight_avg_cost') # 使用已经从 context 更新的平均成本

        # 检查所有需要的数据是否存在
        if not all([cost_95pct, cost_5pct, cost_85pct, cost_15pct, weight_avg_cost]) or weight_avg_cost <= 0:
            return {
                'concentration_90pct': None,
                'concentration_70pct': None,
            }
        # 计算90%筹码的区间宽度
        width_90 = cost_95pct - cost_5pct
        # 计算70%筹码的区间宽度
        width_70 = cost_85pct - cost_15pct
        # 更新上下文中的70%成本区间，供下游使用
        self.ctx['cost_range_70pct'] = (cost_15pct, cost_85pct)
        return {
            'concentration_90pct': width_90 / weight_avg_cost,
            'concentration_70pct': width_70 / weight_avg_cost,
        }

    def _calculate_winner_structure(self) -> dict:
        """
        【V13.6 · 普罗米修斯之火 · 最终正确版】
        - 核心修正: 撤销 V13.4/V13.5 的错误逻辑回归。本版为经过循环论证后的最终正确实现。
        - 根本性原则: “职责分离，各司其职”。承认不同模块对“长/短期”概念有不同业务诉求。
        - 新逻辑:
            1. 本方法旨在分析【当前】盈亏结构，故采用 `weight_avg_cost` (市场重心)作为划分标准，
               彻底规避 `prev_20d_close` 带来的“四象限盲区”问题。
            2. 沿用“宙斯之盾”的二次切分结构，先精确划分“获利/套牢”两大阵营，再在阵营内部
               使用 `weight_avg_cost` 标准进行“高成本区/低成本区”的细分，确保指标的内在逻辑自洽。
        - 收益: 此版本结合了动态划分的灵活性和二次切分的严谨性，是目前已知最健壮、最符合
                业务直觉的实现，能够为下游衍生指标提供最纯净、最可靠的数据源。
        """
        close_price = self.ctx.get('close_price')
        weight_avg_cost = self.ctx.get('weight_avg_cost') # 重新确立 weight_avg_cost 为动态划分基准
        total_winner_rate = self.ctx.get('total_winner_rate', 0.0)
        
        if not all(pd.notna(v) for v in [close_price, weight_avg_cost]):
            return {
                'winner_rate_short_term': None, 'winner_rate_long_term': None,
                'loser_rate_short_term': None, 'loser_rate_long_term': None,
                'total_loser_rate': 100.0 - total_winner_rate if total_winner_rate is not None else None
            }
        
        # 采用“先盈亏，后成本分区”的二次切分法
        # 1. 精确划分获利盘(winners_df)与套牢盘(losers_df)两大阵营
        winners_df = self.df[self.df['price'] < close_price]
        losers_df = self.df[self.df['price'] > close_price]

        # 2. 在各自阵营内部，再使用 `weight_avg_cost` 进行“高成本区/低成本区”细分
        #    这里的“short_term”和“long_term”是成本概念的代理，而非时间概念
        # 在获利盘内部划分
        long_term_winner_rate = winners_df[winners_df['price'] < weight_avg_cost]['percent'].sum()
        short_term_winner_rate = winners_df[winners_df['price'] >= weight_avg_cost]['percent'].sum()

        # 在套牢盘内部划分
        long_term_loser_rate = losers_df[losers_df['price'] < weight_avg_cost]['percent'].sum()
        short_term_loser_rate = losers_df[losers_df['price'] >= weight_avg_cost]['percent'].sum()

        # 3. 计算总套牢盘比例
        total_loser_rate = long_term_loser_rate + short_term_loser_rate
        
        # print(f"DEBUG: trade_time={self.ctx.get('trade_time')}, close={close_price}, avg_cost={weight_avg_cost:.2f}, total_winner_rate={total_winner_rate:.2f}, calculated_winners_sum={(long_term_winner_rate + short_term_winner_rate):.2f}")
        

        return {
            'winner_rate_short_term': short_term_winner_rate,
            'winner_rate_long_term': long_term_winner_rate,
            'loser_rate_short_term': short_term_loser_rate,
            'loser_rate_long_term': long_term_loser_rate,
            'total_loser_rate': total_loser_rate,
        }

    def _calculate_holder_costs(self) -> dict:
        """
        【V1.1 · 阿瑞斯之盾协议】计算长/短期持仓者平均成本
        - 核心修复: 解决了当某一类（长期/短期）筹码不存在时，对应成本指标返回None，
                    从而引发下游 `cost_divergence` 指标“空值雪崩”的根本性问题。
        - 新逻辑: 当某一类筹码为空时，其平均成本被逻辑锚定在划分边界 `prev_20d_close` 上，
                  确保指标始终返回一个有业务意义的有效值，保证了数据计算链的完整性。
        """
        prev_20d_close = self.ctx.get('prev_20d_close')
        if pd.isna(prev_20d_close):
            return {'avg_cost_short_term': None, 'avg_cost_long_term': None}
        # 长期持仓者：成本低于20日前收盘价的筹码
        long_term_chips_df = self.df[self.df['price'] < prev_20d_close]
        # 短期持仓者：成本高于等于20日前收盘价的筹码
        short_term_chips_df = self.df[self.df['price'] >= prev_20d_close]
        # [代码修改开始]
        # 如果长期筹码存在，则计算其加权平均成本
        if not long_term_chips_df.empty and long_term_chips_df['percent'].sum() > 0:
            avg_cost_long = np.average(long_term_chips_df['price'], weights=long_term_chips_df['percent'])
        else:
            # 如果不存在长期筹码，则其成本逻辑上锚定在划分边界
            # print(f"DEBUG: trade_time={self.ctx.get('trade_time')}, 无长期筹码，avg_cost_long_term 锚定为 prev_20d_close: {prev_20d_close}")
            avg_cost_long = prev_20d_close
        # 如果短期筹码存在，则计算其加权平均成本
        if not short_term_chips_df.empty and short_term_chips_df['percent'].sum() > 0:
            avg_cost_short = np.average(short_term_chips_df['price'], weights=short_term_chips_df['percent'])
        else:
            # 如果不存在短期筹码，则其成本逻辑上锚定在划分边界
            # print(f"DEBUG: trade_time={self.ctx.get('trade_time')}, 无短期筹码，avg_cost_short_term 锚定为 prev_20d_close: {prev_20d_close}")
            avg_cost_short = prev_20d_close
        
        return {
            'avg_cost_short_term': avg_cost_short,
            'avg_cost_long_term': avg_cost_long,
        }

    def _calculate_pressure_support(self) -> dict:
        # 上方压力与下方支撑计算：此部分逻辑一直正确，无需修改
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
        # 有效换手计算：此部分逻辑一直正确，无需修改
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
        """
        【V6.2 · 赫尔墨斯信使协议】
        - 核心修复: 修正了 `winner_profit_margin` 的计算逻辑。当不存在获利盘时，
                    明确将其值赋为 0.0，而不是 None。此举解决了因其为空值而导致的
                    下游所有斜率、加速度指标“雪崩式”变为空值的根本性问题。
        - 健壮性增强: 为 `peak_absorption_intensity` 的计算增加了分母为零的保护，
                      避免了在成交量为零的罕见情况下出现除零错误。
        """
        results = {}
        # --- 1. 控盘度指标 (Control Metrics) ---
        peak_volume = context.get('peak_volume')
        total_float_share = self.ctx.get('total_chip_volume')
        if peak_volume is not None and total_float_share and total_float_share > 0:
            results['peak_control_ratio'] = (peak_volume / total_float_share) * 100
        peak_cost = context.get('peak_cost')
        daily_turnover = self.ctx.get('daily_turnover_volume')
        peak_range_low = context.get('peak_range_low')
        peak_range_high = context.get('peak_range_high')
        # 增加分母 daily_turnover 的健壮性检查
        if daily_turnover and daily_turnover > 0 and peak_range_low is not None and peak_range_high is not None:
            turnover_in_peak_range = self._get_turnover_in_range(peak_range_low, peak_range_high)
            # 增加除零保护
            results['peak_absorption_intensity'] = turnover_in_peak_range / daily_turnover if daily_turnover > 0 else 0.0
        else:
            results['peak_absorption_intensity'] = 0.0 # 如果基础数据不全，吸筹强度为0
        
        # --- 2. 利润质量指标 (Profit Quality Metrics) ---
        close_price = self.ctx.get('close_price')
        if close_price:
            winners_df = self.df[self.df['price'] < close_price]
            # 修正获利盘为空时的处理逻辑
            if not winners_df.empty and winners_df['percent'].sum() > 0:
                winner_avg_cost = np.average(winners_df['price'], weights=winners_df['percent'])
                results['winner_avg_cost'] = winner_avg_cost
                if winner_avg_cost > 0:
                    results['winner_profit_margin'] = ((close_price - winner_avg_cost) / winner_avg_cost) * 100
                else:
                    # 获利盘平均成本为0的罕见情况
                    results['winner_profit_margin'] = 0.0
            else:
                # 当不存在获利盘时，平均成本无意义，但利润安全垫的业务含义是0
                results['winner_avg_cost'] = None
                results['winner_profit_margin'] = 0.0
            
        # --- 3. 价码关系指标 (Price-Chip Relation Metrics) ---
        if close_price and peak_cost and peak_cost > 0:
            results['price_to_peak_ratio'] = close_price / peak_cost
        if close_price:
            weighted_mean = self.ctx.get('weight_avg_cost')
            if weighted_mean is not None: # 增加对 weight_avg_cost 的检查
                weighted_std = np.sqrt(np.average((self.df['price'] - weighted_mean)**2, weights=self.df['percent']))
                if weighted_std and weighted_std > 0:
                    results['chip_zscore'] = (close_price - weighted_mean) / weighted_std
        return results

    def _calculate_turnover_structure(self) -> dict:
        """
        【V3.1 · 赫尔墨斯的交易之尺 · 健壮性终极版】计算成交量微观结构
        - 核心修复: 彻底解决了因上游行情数据缺失导致方法返回空字典，从而引发
                    下游衍生指标“空值雪崩”的根本性BUG。
        - 新逻辑: 无论输入如何，此方法都保证返回一个包含
                  'turnover_from_winners_ratio' 和 'turnover_from_losers_ratio'
                  键的字典。在无法计算时，赋予中性值 50.0，确保数据链的完整性。
        """
        close_price = self.ctx.get('close_price')
        low_price = self.ctx.get('low_price')
        high_price = self.ctx.get('high_price')
        # 修正入口保护逻辑，不再返回空字典
        default_return = {
            'turnover_from_winners_ratio': 50.0,
            'turnover_from_losers_ratio': 50.0,
        }
        if not all(pd.notna(v) for v in [close_price, low_price, high_price]):
            # print(f"DEBUG: trade_time={self.ctx.get('trade_time')}, 因行情数据不完整，成交结构返回默认值。")
            return default_return
        
        # 1. 找出当日价格波动区间内所有的筹码
        turnover_zone_df = self.df[(self.df['price'] >= low_price) & (self.df['price'] <= high_price)]
        if turnover_zone_df.empty:
            # 如果当天是“一字板”，则无法按此逻辑计算，返回中性值
            return default_return
        # 2. 在这个“成交活跃区”内，进一步划分获利盘和套牢盘
        winners_in_zone_df = turnover_zone_df[turnover_zone_df['price'] < close_price]
        losers_in_zone_df = turnover_zone_df[turnover_zone_df['price'] > close_price]
        # 3. 计算各自在“成交活跃区”内的筹码占比
        total_percent_in_zone = turnover_zone_df['percent'].sum()
        if total_percent_in_zone == 0:
            return default_return
        winner_contribution_ratio = winners_in_zone_df['percent'].sum() / total_percent_in_zone
        loser_contribution_ratio = losers_in_zone_df['percent'].sum() / total_percent_in_zone
        # 4. 将这个贡献比例作为最终的成交结构比例
        turnover_from_winners_ratio = winner_contribution_ratio * 100
        turnover_from_losers_ratio = loser_contribution_ratio * 100
        return {
            'turnover_from_winners_ratio': turnover_from_winners_ratio,
            'turnover_from_losers_ratio': turnover_from_losers_ratio,
        }

    def _calculate_turnover_at_peak(self, peaks_info: dict) -> dict:
        """
        【V1.0 新增】计算主峰成交占比。
        估算当日总成交量中，发生在主筹码峰价格区间的比例。
        """
        daily_turnover_vol = self.ctx.get('daily_turnover_volume')
        peak_range_low = peaks_info.get('peak_range_low')
        peak_range_high = peaks_info.get('peak_range_high')

        # 检查所有必要数据是否存在且有效
        if not all([daily_turnover_vol, peak_range_low, peak_range_high]) or daily_turnover_vol == 0:
            return {'turnover_at_peak_ratio': None}

        # 调用复用性极高的 _get_turnover_in_range 辅助函数来估算成交量
        turnover_in_peak_range = self._get_turnover_in_range(peak_range_low, peak_range_high)
        
        # 计算比例并转换为百分比
        ratio = turnover_in_peak_range / daily_turnover_vol
        turnover_at_peak_ratio = ratio * 100

        return {
            'turnover_at_peak_ratio': turnover_at_peak_ratio
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
        """
        【V11.1 健壮性修复版】计算筹码断层指标。
        - 核心修复: 解决了 chip_fault_vacuum_percent 字段在不满足“断层”条件时被设为 None 的问题。
                    当价格未显著脱离成本区，即“真空区”不存在时，其筹码占比现在被合理地设为 0.0，
                    而不是 None，从而避免了数据库中出现空值，并方便后续的衍生指标计算。
        """
        results = {
            'chip_fault_strength': None,
            'chip_fault_vacuum_percent': 0.0, # 新增-将默认值从None改为0.0
            'is_chip_fault_formed': False
        }
        peak_cost = context.get('peak_cost')
        close_price = self.ctx.get('close_price')
        if not all([peak_cost, close_price]):
            return results
        # 1. 计算断层强度 (Fault Strength)
        fault_strength = (close_price - peak_cost) / peak_cost if peak_cost > 0 else 0
        results['chip_fault_strength'] = fault_strength
        # 2. 识别断层真空区 (Fault Vacuum)
        fault_zone_low = peak_cost * 1.01
        fault_zone_high = close_price * 0.99
        if fault_zone_high > fault_zone_low:
            fault_zone_df = self.df[(self.df['price'] >= fault_zone_low) & (self.df['price'] <= fault_zone_high)]
            vacuum_chip_percent = fault_zone_df['percent'].sum()
            results['chip_fault_vacuum_percent'] = vacuum_chip_percent
        else:
            # 【代码修改】当真空区不存在时，其筹码占比应为0，而不是None。
            results['chip_fault_vacuum_percent'] = 0.0
        # 3. 最终断层信号 (Fault Signal)
        is_strong_fault = fault_strength > 0.20
        vacuum_percent = results.get('chip_fault_vacuum_percent')
        if vacuum_percent is not None:
            is_vacuum_clear = vacuum_percent < 5.0
        else:
            is_vacuum_clear = False
        results['is_chip_fault_formed'] = is_strong_fault and is_vacuum_clear
        return results





