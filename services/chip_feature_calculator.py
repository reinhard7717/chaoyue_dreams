# 文件: services/calculators/chip_feature_calculator.py

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
        
        for key in ['total_chip_volume', 'daily_turnover_volume', 'weight_avg_cost', 'close_price', 'high_price', 'low_price', 'prev_20d_close']:
            if key in self.ctx and isinstance(self.ctx[key], Decimal):
                self.ctx[key] = float(self.ctx[key])

    def calculate_all_metrics(self) -> dict:
        # 主计算函数：按顺序调用所有子计算模块
        if self.df.empty or not all(k in self.ctx for k in ['weight_avg_cost', 'close_price', 'total_chip_volume']):
            return {}

        peaks_info = self._calculate_peaks()
        concentration_info = self._calculate_concentration()
        winner_structure_info = self._calculate_winner_structure()
        pressure_support_info = self._calculate_pressure_support()
        turnover_info = self._calculate_effective_turnover()
        fund_flow_info = self._calculate_fund_flow_metrics()
        fault_info = self._calculate_chip_fault(context_for_advanced)
        
        # 将之前计算的结果作为输入，传递给新的计算单元
        context_for_advanced = {**self.ctx, **peaks_info, **concentration_info, **winner_structure_info}
        advanced_structure_info = self._calculate_advanced_structures(context_for_advanced)

        return {
            **peaks_info,
            **concentration_info,
            **winner_structure_info,
            **pressure_support_info,
            **turnover_info,
            **fund_flow_info,
            **fault_info,
            **advanced_structure_info # 合并最终结果
        }

    def _calculate_peaks(self) -> dict:
        # 筹码峰计算：此部分逻辑一直正确，无需修改
        peaks, properties = find_peaks(self.df['percent'], prominence=0.1, width=1)
        
        if len(peaks) == 0:
            peak_idx = self.df['percent'].idxmax()
            return {
                'peak_cost': self.df.loc[peak_idx, 'price'],
                'peak_percent': self.df.loc[peak_idx, 'percent'],
                'peak_volume': int(self.df.loc[peak_idx, 'percent'] / 100 * self.ctx['total_chip_volume']),
                'peak_stability': 1.0,
                'is_multi_peak': False,
                'secondary_peak_cost': None,
                'peak_distance_ratio': None,
                'peak_strength_ratio': None,
            }

        prominences = properties['prominences']
        main_peak_idx_in_peaks = np.argmax(prominences)
        main_peak_df_idx = peaks[main_peak_idx_in_peaks]
        
        main_peak_cost = self.df.iloc[main_peak_df_idx]['price']
        main_peak_percent = self.df.iloc[main_peak_df_idx]['percent']
        main_peak_volume = int(main_peak_percent / 100 * self.ctx['total_chip_volume'])
        peak_stability = prominences[main_peak_idx_in_peaks] / self.df['percent'].mean() if self.df['percent'].mean() > 0 else 1.0

        result = {
            'peak_cost': main_peak_cost,
            'peak_percent': main_peak_percent,
            'peak_volume': main_peak_volume,
            'peak_stability': peak_stability,
            'is_multi_peak': len(peaks) > 1,
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

    def _calculate_concentration(self) -> dict:
        # 筹码集中度计算：此部分逻辑一直正确，无需修改
        self.df['cumulative_percent'] = self.df['percent'].cumsum()
        
        def get_concentration_range_vectorized(target_pct: float) -> tuple:
            cum_percent_vals = self.df['cumulative_percent'].values
            price_vals = self.df['price'].values
            
            start_cum_vals = np.roll(cum_percent_vals, 1)
            start_cum_vals[0] = 0
            target_cum_vals = start_cum_vals + target_pct

            end_indices = np.searchsorted(cum_percent_vals, target_cum_vals, side='right')

            valid_mask = end_indices < len(price_vals)
            if not np.any(valid_mask):
                return float('inf'), (None, None)
            
            start_indices_valid = np.arange(len(price_vals))[valid_mask]
            end_indices_valid = end_indices[valid_mask]

            widths = price_vals[end_indices_valid] - price_vals[start_indices_valid]
            
            min_width_idx = np.argmin(widths)
            
            min_width = widths[min_width_idx]
            best_start_price = price_vals[start_indices_valid[min_width_idx]]
            best_end_price = price_vals[end_indices_valid[min_width_idx]]
            
            return min_width, (best_start_price, best_end_price)

        width_90, _ = get_concentration_range_vectorized(90.0)
        _, range_70 = get_concentration_range_vectorized(70.0)
        
        self.ctx['cost_range_70pct'] = range_70

        return {
            'concentration_90pct': width_90 / self.ctx['weight_avg_cost'] if width_90 != float('inf') and self.ctx['weight_avg_cost'] > 0 else None,
        }

    # ▼▼▼【代码修改 V10.0】: 修正获利盘计算逻辑 ▼▼▼
    def _calculate_winner_structure(self) -> dict:
        """
        【V10.0 最终修正】
        严格按照模型字段的业务定义进行计算，并增加总获利盘指标。
        """
        close_price = self.ctx.get('close_price')
        prev_20d_close = self.ctx.get('prev_20d_close')

        # 如果缺少关键价格数据，则无法计算
        if not close_price or pd.isna(prev_20d_close):
            return {'winner_rate_short_term': None, 'winner_rate_long_term': None, 'total_winner_rate': None}

        # 定义1: 总获利盘 = 持仓成本 < 当天收盘价
        total_winners_df = self.df[self.df['price'] < close_price]
        total_winner_rate = total_winners_df['percent'].sum()

        # 定义2: 长期锁定盘 = 持仓成本 < 20日前收盘价
        long_term_winners_df = self.df[self.df['price'] < prev_20d_close]
        long_term_winner_rate = long_term_winners_df['percent'].sum()

        # 定义3: 短期获利盘 = 成本低于收盘价，但高于20日前收盘价
        short_term_winners_df = self.df[(self.df['price'] < close_price) & (self.df['price'] >= prev_20d_close)]
        short_term_winner_rate = short_term_winners_df['percent'].sum()
        
        return {
            'winner_rate_short_term': short_term_winner_rate,
            'winner_rate_long_term': long_term_winner_rate,
            'total_winner_rate': total_winner_rate,
        }
    # ▲▲▲【代码修改 V10.0】▲▲▲

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

    def _calculate_fund_flow_metrics(self) -> dict:
        """
        【V6.1 全天候作战版】
        - 核心升级: 采用弹性融合逻辑，智能处理外部数据源缺失问题。
        - 算法:
          1. 永远以最可靠的交易所数据(CY)为基准。
          2. 检查同花顺(THS)和东方财富(DC)数据是否有效(非None,非NaN)。
          3. 统计有效数据源的数量(1-3个)。
          4. 基于有效数据源，重新定义“共识”与“分歧”：
             - 共识: 至少有2个有效源，且它们的判断完全一致。
             - 分歧: 至少有2个有效源，且它们的判断不一致。
             - 单一信源: 只有1个有效源时，不产生共识或分歧信号。
        """
        # --- 1. 内部评估 (基准数据) ---
        cy_keys = [
            'buy_lg_amount', 'buy_elg_amount', 'sell_lg_amount', 'sell_elg_amount',
            'buy_sm_vol', 'buy_md_vol', 'sell_sm_vol', 'sell_md_vol'
        ]
        # 如果连最核心的交易所数据都没有，则直接放弃
        if not all(pd.notna(self.ctx.get(key)) for key in cy_keys):
            trade_date_str = self.ctx.get('trade_time', '未知日期')
            print(f"调试信息: [{trade_date_str}] 缺少核心交易所资金流数据，跳过所有资金流指标计算。")
            return {}

        # 计算内部评估结果
        main_force_buy_amount = float(self.ctx.get('buy_lg_amount', 0)) + float(self.ctx.get('buy_elg_amount', 0))
        main_force_sell_amount = float(self.ctx.get('sell_lg_amount', 0)) + float(self.ctx.get('sell_elg_amount', 0))
        internal_main_force_net_amount = (main_force_buy_amount - main_force_sell_amount) * 10000

        retail_buy_vol = float(self.ctx.get('buy_sm_vol', 0)) + float(self.ctx.get('buy_md_vol', 0))
        retail_sell_vol = float(self.ctx.get('sell_sm_vol', 0)) + float(self.ctx.get('sell_md_vol', 0))
        internal_retail_net_volume = (retail_buy_vol - retail_sell_vol) * 100

        # --- 2. 外部参照与有效性检查 ---
        # 获取外部数据，如果不存在或为NaN，则为None
        ths_net_amount = self.ctx.get('ths_buy_lg_amount')
        dc_net_amount = self.ctx.get('dc_net_amount')
        
        # 检查数据有效性 (pd.notna可以正确处理None和NaN)
        is_cy_valid = True # 运行到这里，CY必然有效
        is_ths_valid = pd.notna(ths_net_amount)
        is_dc_valid = pd.notna(dc_net_amount)

        # --- 3. 弹性融合裁决 ---
        # 存储所有有效源的判断符号 (-1, 0, 1)
        valid_signs = []
        if is_cy_valid:
            valid_signs.append(np.sign(internal_main_force_net_amount))
        if is_ths_valid:
            valid_signs.append(np.sign(float(ths_net_amount)))
        if is_dc_valid:
            valid_signs.append(np.sign(float(dc_net_amount)))
        
        source_count = len(valid_signs)
        consensus_inflow = False
        consensus_outflow = False
        divergence = False

        # 只有当有效数据源大于等于2个时，才可能产生“共识”或“分歧”
        if source_count >= 2:
            # 检查所有有效源的判断是否完全一致
            is_all_same = all(s == valid_signs[0] for s in valid_signs)
            
            if is_all_same:
                # 如果判断一致，则根据方向确定是共识流入还是共识流出
                if valid_signs[0] > 0:
                    consensus_inflow = True
                elif valid_signs[0] < 0:
                    consensus_outflow = True
            else:
                # 如果判断不一致，则标记为分歧
                divergence = True

        return {
            # 内部评估结果
            'main_force_net_inflow_amount': internal_main_force_net_amount,
            'retail_net_inflow_volume': internal_retail_net_volume,
            # 外部参照数据 (即使无效也存储None)
            'ths_main_force_net_amount': float(ths_net_amount) if is_ths_valid else None,
            'dc_main_force_net_amount': float(dc_net_amount) if is_dc_valid else None,
            # 最终融合信号
            'fund_flow_data_source_count': source_count,
            'consensus_main_force_inflow': consensus_inflow,
            'consensus_main_force_outflow': consensus_outflow,
            'fund_flow_divergence': divergence,
        }

    def _calculate_advanced_structures(self, context: dict) -> dict:
        """
        【V6.0 新增】计算“控盘度”、“利润质量”、“价码关系”三大升维指标。
        """
        results = {}
        
        # --- 1. 控盘度指标 (Control Metrics) ---
        peak_volume = context.get('peak_volume')
        total_float_share = self.ctx.get('total_chip_volume') # total_chip_volume 就是流通股本
        if peak_volume is not None and total_float_share and total_float_share > 0:
            results['peak_control_ratio'] = (peak_volume / total_float_share) * 100

        # 计算筹码峰吸筹强度需要筹码峰的宽度信息
        # (这是一个简化实现，更精确的需要从find_peaks获取宽度)
        peak_cost = context.get('peak_cost')
        daily_turnover = self.ctx.get('daily_turnover_volume')
        if peak_cost is not None and daily_turnover and daily_turnover > 0:
            # 假设主峰的核心范围是其成本价的上下1%
            peak_range_low = peak_cost * 0.99
            peak_range_high = peak_cost * 1.01
            # 估算在主峰范围内的换手比例
            turnover_in_peak_range = self._get_turnover_in_range(peak_range_low, peak_range_high)
            results['peak_absorption_intensity'] = turnover_in_peak_range / daily_turnover if daily_turnover > 0 else 0

        # --- 2. 利润质量指标 (Profit Quality Metrics) ---
        close_price = self.ctx.get('close_price')
        if close_price:
            winners_df = self.df[self.df['price'] < close_price]
            if not winners_df.empty:
                # 计算获利盘的加权平均成本
                winner_avg_cost = np.average(winners_df['price'], weights=winners_df['percent'])
                results['winner_avg_cost'] = winner_avg_cost
                # 计算获利盘的安全垫
                if winner_avg_cost > 0:
                    results['winner_profit_margin'] = ((close_price - winner_avg_cost) / winner_avg_cost) * 100

        # --- 3. 价码关系指标 (Price-Chip Relation Metrics) ---
        if close_price and peak_cost and peak_cost > 0:
            results['price_to_peak_ratio'] = close_price / peak_cost
        
        # 计算筹码Z-Score
        if close_price:
            weighted_mean = self.ctx.get('weight_avg_cost')
            # 计算加权标准差
            weighted_std = np.sqrt(np.average((self.df['price'] - weighted_mean)**2, weights=self.df['percent']))
            if weighted_std and weighted_std > 0:
                results['chip_zscore'] = (close_price - weighted_mean) / weighted_std

        # --- 4. 超级指标: 筹码健康分 (Chip Health Score) ---
        # 这是一个可以持续优化的专家打分系统，这里提供一个初始版本
        score = 50.0 # 基础分
        # 集中度越高越好
        conc_90 = context.get('concentration_90pct', 1.0)
        score += (0.3 - min(conc_90, 0.3)) * 100 # 集中度低于30%开始加分，最多加30分
        # 集中度在收敛加分
        conc_slope = context.get('concentration_90pct_slope_5d', 0)
        if conc_slope < 0: score += 5
        # 获利盘安全垫越高越好
        profit_margin = results.get('winner_profit_margin', 0)
        score += min(profit_margin, 20) # 最多加20分
        # 股价在主峰之上加分
        price_ratio = results.get('price_to_peak_ratio', 1.0)
        if price_ratio > 1.05: score += 10
        # 共识性流入加分
        if context.get('consensus_main_force_inflow'): score += 15
        # 共识性流出扣分
        if context.get('consensus_main_force_outflow'): score -= 20
        
        results['chip_health_score'] = max(0, min(100, round(score, 2))) # 确保分数在0-100之间

        return results

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
        【V11.0 新增】计算筹码断层指标。
        识别股价脱离核心成本区后形成的“真空地带”。
        """
        results = {}
        peak_cost = context.get('peak_cost')
        close_price = self.ctx.get('close_price')

        if not all([peak_cost, close_price]):
            return results

        # 1. 计算断层强度 (Fault Strength)
        # 定义：当前价格脱离主筹码峰的程度
        fault_strength = (close_price - peak_cost) / peak_cost if peak_cost > 0 else 0
        results['chip_fault_strength'] = fault_strength

        # 2. 识别断层真空区 (Fault Vacuum)
        # 定义：主筹码峰与当前价格之间，筹码的稀疏程度
        # 选取主峰上方1%到收盘价下方1%的区间作为“断层带”
        fault_zone_low = peak_cost * 1.01
        fault_zone_high = close_price * 0.99
        
        if fault_zone_high > fault_zone_low:
            fault_zone_df = self.df[(self.df['price'] >= fault_zone_low) & (self.df['price'] <= fault_zone_high)]
            # 真空区的筹码占比，越低越好
            vacuum_chip_percent = fault_zone_df['percent'].sum()
            results['chip_fault_vacuum_percent'] = vacuum_chip_percent
        else:
            results['chip_fault_vacuum_percent'] = 0 # 如果没有空间，则真空度为0

        # 3. 最终断层信号 (Fault Signal)
        # 定义：断层强度足够大（如脱离成本区20%以上），且真空区足够“空”（如筹码占比低于5%）
        is_strong_fault = fault_strength > 0.20
        is_vacuum_clear = results.get('chip_fault_vacuum_percent', 100) < 5.0
        results['is_chip_fault_formed'] = is_strong_fault and is_vacuum_clear

        return results





